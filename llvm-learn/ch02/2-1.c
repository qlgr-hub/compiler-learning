int add(int a, int b) {
    return a + b;
}


// clang -cc1 -ast-dump 2-1.c or clang -Xclang -ast-dump 2-1.c 
// TranslationUnitDecl 0x6360f7815ea8 <<invalid sloc>> <invalid sloc>
// |-TypedefDecl 0x6360f78166d8 <<invalid sloc>> <invalid sloc> implicit __int128_t '__int128'
// | `-BuiltinType 0x6360f7816470 '__int128'
// |-TypedefDecl 0x6360f7816748 <<invalid sloc>> <invalid sloc> implicit __uint128_t 'unsigned __int128'
// | `-BuiltinType 0x6360f7816490 'unsigned __int128'
// |-TypedefDecl 0x6360f7816a50 <<invalid sloc>> <invalid sloc> implicit __NSConstantString 'struct __NSConstantString_tag'
// | `-RecordType 0x6360f7816820 'struct __NSConstantString_tag'
// |   `-Record 0x6360f78167a0 '__NSConstantString_tag'
// |-TypedefDecl 0x6360f7816af8 <<invalid sloc>> <invalid sloc> implicit __builtin_ms_va_list 'char *'
// | `-PointerType 0x6360f7816ab0 'char *'
// |   `-BuiltinType 0x6360f7815f50 'char'
// |-TypedefDecl 0x6360f7816df0 <<invalid sloc>> <invalid sloc> implicit __builtin_va_list 'struct __va_list_tag[1]'
// | `-ConstantArrayType 0x6360f7816d90 'struct __va_list_tag[1]' 1 
// |   `-RecordType 0x6360f7816bd0 'struct __va_list_tag'
// |     `-Record 0x6360f7816b50 '__va_list_tag'
// `-FunctionDecl 0x6360f7868608 <2-1.c:1:1, line:3:1> line:1:5 add 'int (int, int)'
//   |-ParmVarDecl 0x6360f78684a0 <col:9, col:13> col:13 used a 'int'
//   |-ParmVarDecl 0x6360f7868520 <col:16, col:20> col:20 used b 'int'
//   `-CompoundStmt 0x6360f78687a8 <col:23, line:3:1>
//     `-ReturnStmt 0x6360f7868798 <line:2:5, col:16>
//       `-BinaryOperator 0x6360f7868778 <col:12, col:16> 'int' '+'
//         |-ImplicitCastExpr 0x6360f7868748 <col:12> 'int' <LValueToRValue>
//         | `-DeclRefExpr 0x6360f7868708 <col:12> 'int' lvalue ParmVar 0x6360f78684a0 'a' 'int'
//         `-ImplicitCastExpr 0x6360f7868760 <col:16> 'int' <LValueToRValue>
//           `-DeclRefExpr 0x6360f7868728 <col:16> 'int' lvalue ParmVar 0x6360f7868520 'b' 'int'

