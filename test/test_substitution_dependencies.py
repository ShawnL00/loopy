"""Cached substitution-rule dependency analysis and its expansion fallback."""

from __future__ import annotations

import numpy as np
import pytest

import loopy as lp
from loopy.kernel.tools import (
    InstructionDependencyInfo,
    get_instruction_dependency_info,
)
from loopy.symbolic import (
    get_dependencies,
    get_reduction_inames,
    get_substitution_rule_dependencies,
    parse,
)
from loopy.version import (
    LOOPY_USE_LANGUAGE_VERSION_2018_2,  # ruff:ignore[unused-import]
)


def _assert_dependency_info_matches_expansion(
        program: lp.TranslationUnit,
    ) -> None:
    kernel = program.default_entrypoint
    dependency_info = get_instruction_dependency_info(kernel)
    expanded = lp.expand_subst(kernel)
    for insn in expanded.instructions:
        info = dependency_info[insn.id]
        assert info.read_dependency_names == insn.read_dependency_names()
        assert info.reduction_inames == insn.reduction_inames()
        assert info.write_dependency_names == insn.write_dependency_names()


@pytest.mark.parametrize(("domain", "instructions"), [
    pytest.param(
        "{[i,j,k]: 0<=i,j,k<n}",
        """
            base := tmp + a[j]
            used(x) := x + base
            unused(x, y) := used(x)
            nested(x, y) := sum(j, unused(x, y))
            <> tmp = b[i] {id=write}
            out[i] = nested(c[k], sum(j, d[j])) {id=read}
        """,
        id="nested-rules-and-unused-argument"),
    pytest.param(
        "{[i,j]: 0<=i,j<n}",
        """
            # In outer_bound(a[j]), j occurs only inside sum(j, ...), so it is
            # a reduction iname. In outer_mixed(a[j]), the occurrence outside
            # the sum makes j remain a read dependency.
            bound := sum(j, x)
            mixed := x + sum(j, x)
            outer_bound(x) := bound
            outer_mixed(x) := mixed
            out_bound[i] = outer_bound(a[j]) {id=bound}
            out_mixed[i] = outer_mixed(a[j]) {id=mixed}
        """,
        id="bound-and-free-occurrences"),
    pytest.param(
        "{[i,j]: 0<=i,j<n}",
        """
            index(k) := k + offset
            out[index(i)] = a[j]
        """,
        id="rule-in-assignee-index"),
    pytest.param(
        "{[i]: 0<=i<4}",
        """
            inner := x
            outer(x) := inner + x
            out[i] = outer(a[i])
        """,
        id="dynamic-capture"),
    pytest.param(
        "{[i,j]: 0<=i,j<4}",
        """
            inner := sum(j, a[j])
            middle(y) := inner + y
            outer(j) := middle(j)
            out[i] = outer(i)
        """,
        id="binder-renamed-by-caller"),
    pytest.param(
        "{[i]: 0<=i<4}",
        """
            identity(x) := x
            one := 1
            out[out[0]] = identity(out[i]) + a[i] + one
        """,
        id="self-read-on-both-sides"),
])
def test_substitution_rule_dependency_info_matches_expansion(
        domain, instructions):
    program = lp.make_kernel(
        domain,
        instructions,
        silenced_warnings=["inferred_iname"],
    )
    _assert_dependency_info_matches_expansion(program)


def test_kernel_dependency_inference_does_not_call_expand_subst(monkeypatch):
    import loopy.transform.subst

    def fail_expand_subst(*args, **kwargs):
        raise AssertionError("dependency inference called expand_subst")

    monkeypatch.setattr(loopy.transform.subst, "expand_subst", fail_expand_subst)

    program = lp.make_kernel(
            "{[i,j]: 0<=i,j<n}",
            """
                inner(x) := x + tmp
                outer(x) := sum(j, inner(x))

                <> tmp = a[i] {id=write}
                out[i] = outer(b[i]) {id=read}
                """,
            silenced_warnings=["inferred_iname"])

    read = program.default_entrypoint.id_to_insn["read"]
    assert read.depends_on == {"write"}
    assert read.within_inames == {"i"}


def test_repeated_deep_substitution_kernel_creation(monkeypatch):
    import loopy.kernel.creation

    def fail_argument_guessing(*args, **kwargs):
        pytest.fail("explicit kernel arguments triggered argument guessing")

    monkeypatch.setattr(
        loopy.kernel.creation, "ArgumentGuesser", fail_argument_guessing)

    rules = ["level_0(x) := x + 1"]
    for level in range(1, 15):
        rules.append(
            f"level_{level}(x) := "
            f"level_{level - 1}(x + 1) + level_{level - 1}(x + 2)"
        )

    source = "\n".join([
        *rules,
        "out[i] = level_14(a[i]) {id=write}",
    ])
    kernel_data = [
        lp.GlobalArg("a", np.float64, shape=(4,)),
        lp.GlobalArg("out", np.float64, shape=(4,), is_input=False),
    ]

    first = lp.make_kernel("{[i]: 0 <= i < 4}", source, kernel_data)
    second = lp.make_kernel("{[i]: 0 <= i < 4}", source, kernel_data)

    assert first.default_entrypoint.instructions == (
        second.default_entrypoint.instructions
    )


def test_substitution_rule_alias_as_reduction_iname():
    program = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}",
        """
            alias_0 := j
            alias := alias_0
            row_sum(row) := sum(alias, a[row, alias])
            out[i] = row_sum(i) {id=write}
        """,
        [
            lp.GlobalArg("a", np.float64, shape=(4, 4)),
            lp.GlobalArg("out", np.float64, shape=(4,), is_input=False),
        ],
        target=lp.CTarget(),
        silenced_warnings=["inferred_iname"],
    )

    _assert_dependency_info_matches_expansion(program)
    assert program.default_entrypoint.id_to_insn["write"].within_inames == {"i"}
    lp.generate_code_v2(program).device_code()


def test_substitution_rule_dependency_info_for_call_instruction():
    rules = {
        "alias": lp.SubstitutionRule("alias", (), parse("j")),
        "one": lp.SubstitutionRule("one", (), parse("1")),
        "view": lp.SubstitutionRule(
            "view", ("row",), parse("[alias]: a[row, alias]")),
    }
    program = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}",
        [lp.CallInstruction(
            (parse("[alias]: out[out[0], alias]"),),
            parse("f(view(i), one$tag)"),
            id="call")],
        substitutions=rules,
        kernel_data=[
            lp.GlobalArg("a", np.float64, shape=(4, 4)),
            lp.GlobalArg("out", np.float64, shape=(4, 4)),
        ],
    )
    _assert_dependency_info_matches_expansion(program)


def test_substitution_rule_dependency_validation():
    invalid_rule = lp.SubstitutionRule("invalid_rule", (), parse("a"))
    with pytest.raises(
            lp.LoopyError,
            match="must be a function call"):
        lp.make_kernel(
            [],
            [lp.CallInstruction(
                (), parse("invalid_rule()"), id="invalid_call")],
            substitutions={"invalid_rule": invalid_rule},
            kernel_data="...",
        )


@pytest.mark.parametrize("invalid_expression", ["sum(k, a[k])", "[k]: a[k]"])
@pytest.mark.parametrize("kernel_data", [
    (Ellipsis,),
    ("...",),
    "a,out,n",
    (
        lp.GlobalArg("a", np.float64, shape=("n",)),
        lp.GlobalArg("out", np.float64, shape=("n",)),
        lp.ValueArg("n", np.int32),
    ),
], ids=["ellipsis", "ellipsis-string", "names", "explicit"])
def test_substitution_rule_dependency_rejects_unused_formal_bound_name(
        kernel_data, invalid_expression):
    with pytest.raises(
            lp.LoopyError,
            match="argument 'k' cannot be used as a reduction or swept iname"):
        lp.make_kernel(
            "{[i]: 0<=i<n}",
            f"""
                invalid(k) := {invalid_expression}
                out[i] = a[i]
            """,
            kernel_data,
            silenced_warnings=["inferred_iname"],
        )


def test_substitution_rule_dependency_rejects_formal_reduction_iname():
    from loopy.symbolic import _SubstitutionRuleAwareDependencyMapper

    rules = {
        "rule": lp.SubstitutionRule(
            "rule", ("k",), parse("sum(k, a[k])")),
    }

    with pytest.raises(
            lp.LoopyError,
            match="argument 'k' cannot be used as a reduction or swept iname"):
        _SubstitutionRuleAwareDependencyMapper(rules).get_dependency_info(
            parse("rule(i)"))


def _program():
    return lp.make_kernel(
        "{[i,j]: 0<=i,j<4}",
        """
        base := a[j] + p
        used(x) := x + base
        unused(x,y) := used(x)
        row(x) := sum(j, unused(x, missing))
        identity(x) := x
        index(x) := x + offset
        out[index(i)] = identity(out[i]) + row(b[i]) {id=result}
        """,
        [lp.GlobalArg("out", np.float64, shape=(8,)), "..."],
        lang_version=(2018, 2),
        silenced_warnings=["inferred_iname"],
    )


def test_public_rule_dependencies_match_expansion():
    kernel = _program().default_entrypoint
    dependencies = get_substitution_rule_dependencies(kernel.substitutions)
    assert dependencies["base"] == frozenset({"a", "j", "p"})
    assert dependencies["unused"] == frozenset({"a", "j", "p", "x"})
    assert dependencies["row"] == frozenset({"a", "p", "x"})
    assert all(isinstance(names, frozenset) for names in dependencies.values())
    assert dependencies.keys() == kernel.substitutions.keys()

    # Query the body, not a call: formal arguments must remain symbolic.
    for name, rule in kernel.substitutions.items():
        probe = kernel.copy(instructions=[
            lp.Assignment("probe", rule.expression, id="probe"),
        ])
        expanded = lp.expand_subst(probe).instructions[0].expression
        assert dependencies[name] == get_dependencies(expanded)


def test_public_instruction_dependencies_match_expansion():
    kernel = _program().default_entrypoint
    info = get_instruction_dependency_info(kernel)
    assert info["result"] == InstructionDependencyInfo(
        read_dependency_names=frozenset({"a", "b", "i", "offset", "out", "p"}),
        reduction_inames=frozenset({"j"}),
        write_dependency_names=frozenset({"out", "i", "offset"}),
    )
    assert info.keys() == kernel.id_to_insn.keys()
    for insn in lp.expand_subst(kernel).instructions:
        assert info[insn.id].read_dependency_names == insn.read_dependency_names()
        assert info[insn.id].reduction_inames == insn.reduction_inames()
        assert info[insn.id].write_dependency_names == insn.write_dependency_names()


@pytest.mark.parametrize("swept", [False, True])
def test_public_instruction_dependencies_preserve_assignee_self_index(swept):
    rules = {"one": lp.SubstitutionRule("one", (), parse("1"))}
    if swept:
        instruction = lp.CallInstruction(
            (parse("[j]: out[out[0,0],j]"),),
            parse("f(a[i], one)"), id="result",
        )
        out_shape = (4, 4)
    else:
        instruction = lp.Assignment(
            parse("out[out[0]]"), parse("a[i] + one"), id="result",
        )
        out_shape = (4,)
    kernel = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}", [instruction], substitutions=rules,
        kernel_data=[
            lp.GlobalArg("a", np.int32, shape=(4,)),
            lp.GlobalArg("out", np.int32, shape=out_shape),
        ],
        lang_version=(2018, 2),
    ).default_entrypoint
    info = get_instruction_dependency_info(kernel)["result"]
    expanded, = lp.expand_subst(kernel).instructions
    assert info.read_dependency_names == expanded.read_dependency_names()
    assert info.read_dependency_names == frozenset({"a", "i", "out"})


@pytest.mark.parametrize(("swept_iname", "expected_dependencies"), [
    ("alias", {"a"}),
    ("alias$tag", {"a", "j"}),
])
def test_swept_assignee_alias_preserves_tags(swept_iname, expected_dependencies):
    # A tagged bare name is not a rule reference. Stripping its tag would
    # incorrectly remove the free j in the index from the read dependencies.
    kernel = lp.make_kernel(
        "{[j]: 0<=j<4}",
        [lp.CallInstruction(
            (parse(f"[{swept_iname}]: out[alias]"),),
            parse("f(a)"), id="result",
        )],
        [lp.GlobalArg("out", shape=(4,)), lp.ValueArg("a")],
        substitutions={"alias": lp.SubstitutionRule("alias", (), parse("j"))},
        lang_version=(2018, 2),
        silenced_warnings=["inferred_iname"],
    ).default_entrypoint
    info = get_instruction_dependency_info(kernel)["result"]
    expanded, = lp.expand_subst(kernel).instructions
    assert info.read_dependency_names == expanded.read_dependency_names()
    assert info.read_dependency_names == expected_dependencies


def test_public_dependencies_without_rules():
    kernel = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}", "out[i] = sum(j, a[i,j])",
        [
            lp.GlobalArg("a", np.float64, shape=(4, 4)),
            lp.GlobalArg("out", np.float64, shape=(4,), is_input=False),
        ],
        lang_version=(2018, 2),
    ).default_entrypoint
    assert get_substitution_rule_dependencies({}) == {}
    info = get_instruction_dependency_info(kernel)
    insn, = kernel.instructions
    assert info[insn.id].read_dependency_names == insn.read_dependency_names()
    assert info[insn.id].reduction_inames == get_reduction_inames(insn.expression)


def test_remove_unused_arguments_matches_expansion():
    kernel = _program().default_entrypoint
    kernel = kernel.copy(args=[*kernel.args, lp.ValueArg("unused")])
    expected = lp.remove_unused_arguments(lp.expand_subst(kernel))
    actual = lp.remove_unused_arguments(kernel)
    assert actual.args == expected.args
    assert set(actual.arg_dict) == {"a", "b", "offset", "out", "p"}
    assert actual.instructions == kernel.instructions
    assert actual.substitutions == kernel.substitutions


def test_public_dependency_queries_do_not_expand(monkeypatch):
    import loopy.transform.subst

    rules = ["level_0(x) := x + 1"]
    for level in range(1, 19):
        rules.append(
            f"level_{level}(x) := "
            f"level_{level - 1}(x) + level_{level - 1}(x)"
        )
    program = lp.make_kernel(
        "{[i]: 0<=i<4}",
        [*rules, "out[i] = level_18(a[i]) {id=result}"],
        [
            lp.GlobalArg("a", np.float64, shape=(4,)),
            lp.GlobalArg("out", np.float64, shape=(4,), is_input=False),
        ],
        lang_version=(2018, 2),
    )

    def reject(*args, **kwargs):
        pytest.fail("dependency query expanded substitution rules")

    monkeypatch.setattr(lp, "expand_subst", reject)
    monkeypatch.setattr(loopy.transform.subst, "expand_subst", reject)
    kernel = program.default_entrypoint
    assert set(get_substitution_rule_dependencies(kernel.substitutions).values()) == {
        frozenset({"x"}),
    }
    assert get_instruction_dependency_info(kernel)["result"].read_dependency_names == {
        "a", "i",
    }
    assert lp.remove_unused_arguments(kernel) == kernel


def test_public_rule_query_does_not_reuse_another_graph_cache():
    rules = {"f": lp.SubstitutionRule("f", (), parse("a[i]"))}
    assert get_substitution_rule_dependencies(rules) == {
        "f": frozenset({"a", "i"}),
    }
    rules["f"] = lp.SubstitutionRule("f", (), parse("b[j]"))
    assert get_substitution_rule_dependencies(rules) == {
        "f": frozenset({"b", "j"}),
    }


@pytest.mark.parametrize(("rules", "message"), [
    (
        {
            "f": lp.SubstitutionRule("f", (), parse("g")),
            "g": lp.SubstitutionRule("g", (), parse("f")),
        },
        "recursive substitution rules",
    ),
    (
        {
            "f": lp.SubstitutionRule("f", ("x",), parse("x")),
            "g": lp.SubstitutionRule("g", (), parse("f(a, b)")),
        },
        "number of arguments",
    ),
])
def test_public_rule_query_preserves_invalid_rule_errors(rules, message):
    with pytest.raises(lp.LoopyError, match=message):
        get_substitution_rule_dependencies(rules)


@pytest.mark.parametrize(("rules", "index"), [
    ("", "j"),
    ("index := j", "index"),
    ("index(k) := k", "index(j)"),
    ("inner(k) := k\nindex(k) := inner(k)", "index(j)"),
])
def test_rule_in_writer_index_does_not_propagate_writer_loop(rules, index):
    program = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}",
        f"{rules}\nt[{index}] = a[j] {{id=write}}\n"
        "out[i] = t[0] {id=read}",
        [
            lp.GlobalArg("a", np.float64, shape=(4,)),
            lp.GlobalArg("out", np.float64, shape=(4,)),
            lp.TemporaryVariable("t", np.float64, shape=(4,)),
        ],
        target=lp.CTarget(),
        silenced_warnings=["inferred_iname"],
    )
    kernel = program.default_entrypoint
    info = get_instruction_dependency_info(kernel)
    assert info["write"].write_dependency_names == {"t", "j"}
    assert info["read"].write_dependency_names == {"out", "i"}
    assert kernel.id_to_insn["write"].within_inames == {"j"}
    assert kernel.id_to_insn["read"].within_inames == {"i"}
    assert kernel.id_to_insn["read"].depends_on == {"write"}
    lp.generate_code_v2(program).device_code()


def test_scalar_writer_still_propagates_implicit_inames():
    kernel = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}",
        "value := a[j]\nt = value {id=write}\nout[i] = t {id=read}",
        [
            lp.GlobalArg("a", np.float64, shape=(4,)),
            lp.GlobalArg("out", np.float64, shape=(4,)),
            lp.TemporaryVariable("t", np.float64, shape=()),
        ],
        silenced_warnings=["inferred_iname"],
    ).default_entrypoint
    info = get_instruction_dependency_info(kernel)
    assert info["write"].write_dependency_names == {"t"}
    assert info["write"].read_dependency_names == {"a", "j"}
    assert kernel.id_to_insn["read"].within_inames == {"i", "j"}
    assert kernel.id_to_insn["read"].depends_on == {"write"}


def test_assignment_dependency_fast_path(monkeypatch):
    import loopy.transform.subst

    def reject(*args, **kwargs):
        pytest.fail("ordinary assignment dependency analysis expanded rules")

    monkeypatch.setattr(lp, "expand_subst", reject)
    monkeypatch.setattr(loopy.transform.subst, "expand_subst", reject)
    kernel = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}",
        """
        row(k) := sum(j, sin(a[k,j]))
        index(k) := k + offset
        enabled(k) := mask[k] > 0
        out[index(i)] = row(i) {if=enabled(i), id=result}
        """,
        [
            lp.GlobalArg("a", np.float64, shape=(4, 4)),
            lp.GlobalArg("out", np.float64, shape=(8,), is_input=False),
            lp.GlobalArg("mask", np.int32, shape=(4,)),
            lp.ValueArg("offset", np.int32),
        ],
    ).default_entrypoint

    assert get_instruction_dependency_info(kernel)["result"] == (
        InstructionDependencyInfo(
            read_dependency_names=frozenset({"a", "i", "mask", "offset"}),
            reduction_inames=frozenset({"j"}),
            write_dependency_names=frozenset({"out", "i", "offset"}),
        )
    )
    assert get_substitution_rule_dependencies(kernel.substitutions) == {
        "row": frozenset({"a", "k"}),
        "index": frozenset({"k", "offset"}),
        "enabled": frozenset({"k", "mask"}),
    }


@pytest.mark.parametrize(("definitions", "expression", "reads", "reductions"), [
    pytest.param(
        [("inner", (), "b$site[i]"), ("outer", ("b",), "inner")],
        "outer(a[i])", {"b", "i"}, set(), id="tagged-formal-name"),
    pytest.param(
        [("value", (), "a$site[i]")],
        "value", {"a", "i"}, set(), id="tagged-array"),
    pytest.param(
        [("value", (), "a[i]")],
        "sin$site(value)", {"a", "i"}, set(), id="tagged-call"),
    pytest.param(
        [("view", (), "[j]: a[i,j]")],
        "f(view)", {"a", "i"}, set(), id="sub-array-reference"),
    pytest.param(
        [("inner", (), "sum(j,x) + sum(k,x)"),
         ("outer", ("j", "k", "x"), "inner")],
        "outer(i,i,a[i])", {"a"}, {"i"}, id="merged-reduction-inames"),
])
def test_dependency_fallback_preserves_reads(
        monkeypatch, definitions, expression, reads, reductions):
    import loopy.transform.subst

    kernel = lp.make_kernel(
        "{[i,j,k,m]: 0<=i,j,k,m<4}", "out[m] = 0",
        [lp.GlobalArg("out", shape=(4,))],
    ).default_entrypoint.copy(
        substitutions={
            name: lp.SubstitutionRule(name, arguments, parse(body))
            for name, arguments, body in definitions
        },
        instructions=[lp.Assignment(
            parse("out[m]"), parse(expression), id="result")],
    )
    original_instructions = kernel.instructions
    original_rules = kernel.substitutions
    expand_subst = lp.expand_subst
    expansion_calls = []

    def record_expansion(*args, **kwargs):
        expansion_calls.append(args[0])
        return expand_subst(*args, **kwargs)

    monkeypatch.setattr(lp, "expand_subst", record_expansion)
    monkeypatch.setattr(loopy.transform.subst, "expand_subst", record_expansion)
    info = get_instruction_dependency_info(kernel)["result"]
    assert expansion_calls
    assert info.read_dependency_names == reads | {"m"}
    assert info.reduction_inames == reductions
    assert info.write_dependency_names == {"out", "m"}

    dependencies = get_substitution_rule_dependencies(kernel.substitutions)
    assert all(isinstance(names, frozenset) for names in dependencies.values())
    for name, rule in kernel.substitutions.items():
        probe = kernel.copy(instructions=[
            lp.Assignment("probe", rule.expression, id="probe"),
        ])
        expanded = expand_subst(probe).instructions[0].expression
        assert dependencies[name] == get_dependencies(expanded)
    assert kernel.instructions is original_instructions
    assert kernel.substitutions is original_rules
