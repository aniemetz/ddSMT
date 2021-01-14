import pytest
from ..nodes import Node
from ..smtlib import *


def test_get_variables_with_type():
    x = Node('x')
    y = Node('y')
    z = Node('z')
    r = Node('r')
    s = Node('s')
    a = Node('a')
    b = Node('b')
    c = Node('c')
    exprs = [
        Node('declare-const', x, ('_', 'BitVec', 8)),
        Node('declare-fun', y, (), ('_', 'BitVec', 8)),
        Node('declare-const', z, ('_', 'BitVec', 32)),
        Node('declare-fun', r, (), 'Real'),
        Node('declare-const', s, 'String'),
        Node('declare-const', a, 'Float64'),
        Node('declare-const', b, ('_', 'FloatingPoint', 11, 53)),
        Node('declare-fun', c, (), ('_', 'FloatingPoint', 5, 11)),
        Node('bvadd', x, y),
        Node('+', r, r),
        Node('str.++', s, s),
    ]
    assert get_variables_with_type(Node('_', 'BitVec', 8)) == []
    collect_information(exprs)
    assert get_variables_with_type(Node('_', 'BitVect', 8)) == []
    assert get_variables_with_type(Node('_', 'BitVec', 8)) == [x, y]
    assert get_variables_with_type(Node('_', 'BitVec', 32)) == [z]
    assert get_variables_with_type(Node('Float64')) == [a]
    assert get_variables_with_type(Node('_', 'FloatingPoint', 11, 53)) == [b]
    assert get_variables_with_type(Node('_', 'FloatingPoint', 5, 11)) == [c]
    assert get_variables_with_type(Node('Real')) == [r]
    assert get_variables_with_type(Node('String')) == [s]


def test_introduce_variables():
    fp64 = Node('_', 'FloatingPoint', 11, 53)
    x1 = Node('x1')
    x2 = Node('x2')
    x3 = Node('x3')
    decl_x = Node('declare-const', 'x', 'Real')
    decl_x1 = Node('declare-fun', x1, (), fp64)
    decl_x2 = Node('declare-fun', x2, (), fp64)
    decl_x3 = Node('declare-fun', x3, (), fp64)
    fpadd = Node('define-fun', 'fpadd', (), fp64, ('fp.add', x1, x2))
    fpmul = Node('define-fun', 'fpmul', (), fp64, ('fp.mul', fpadd, x3))
    fpass = Node('assert', ('=', 'fpmul', 'fpadd'))

    exprs0 = [
        Node('set-info', ':smt-lib-version', '2.6'),
        Node('set-logic', 'QF_FP'),
        Node('set-info', ':category', '"crafted"'),
        Node('set-info', ':status', 'unknown'),
    ]
    assert introduce_variables(exprs0, [decl_x]) \
           == exprs0 + [Node('declare-const', 'x', 'Real')]

    exprs1 = exprs0 + [Node('assert', 'true')]
    assert introduce_variables(exprs1, [decl_x]) \
           == exprs0 + [decl_x, Node('assert', 'true')]

    exprs2 = exprs0 + [decl_x1, decl_x2, decl_x3, fpadd, fpmul, fpass]
    assert introduce_variables(exprs2, [decl_x]) \
           == exprs0 + [decl_x1, decl_x2, decl_x3, fpadd, fpmul, decl_x, fpass]


def test_is_indexed_operator():
    assert not is_indexed_operator(Node('x'), 'extract')
    assert not is_indexed_operator(Node('_', 'sign_extend', 2), 'extract')
    assert not is_indexed_operator(Node('extract', 2), 'extract')
    assert not is_indexed_operator(Node(('_', 'extract', 2), 'x'), 'extract')
    assert not is_indexed_operator(Node('_', 'extract', 2, 2), 'extract')
    assert not is_indexed_operator(Node('_', 'extract', 2), 'extract', 2)
    assert is_indexed_operator(Node('_', 'extract', 2), 'extract')


def test_get_bv_constant_value():
    with pytest.raises(AssertionError):
        get_bv_constant_value(Node('x'))
    with pytest.raises(AssertionError):
        get_bv_constant_value(Node('#c3'))
    assert get_bv_constant_value(Node('_', 'bv3', 3)) == (3, 3)
    assert get_bv_constant_value(Node('#b011')) == (3, 3)
    assert get_bv_constant_value(Node('#x3')) == (3, 4)


def test_get_bv_extend_index():
    with pytest.raises(AssertionError):
        get_bv_extend_index(Node('_', 'extend', 2))
    with pytest.raises(ValueError):
        get_bv_extend_index(Node('_', 'zero_extend', 'asdf'))
    assert get_bv_extend_index(Node('_', 'zero_extend', 2)) == 2


# TODO
#def collect_information(exprs):
#def is_leaf(node):
#def is_var(node):
#def has_name(node):
#def get_name(node):
#def is_quoted_symbol(node):
#def get_quoted_symbol(node):
#def is_operator_app(node, name):
#def is_nary(node):
#def is_constant(node):
#def is_eq(node):
#def get_constants(const_type):
#def get_type(node):
#def is_boolean_constant(node):
#def is_arithmetic_constant(node):
#def is_int_constant(node):
#def is_real_constant(node):
#def is_bv_type(node):
#def is_bv_constant(node):
#def is_bv_comp(node):
#def is_bv_not(node):
#def is_bv_neg(node):
#def get_bv_width(node):
#def is_defined_function(node):
#def get_defined_function(node):
#def is_set_type(node):
#def is_string_constant(node):
