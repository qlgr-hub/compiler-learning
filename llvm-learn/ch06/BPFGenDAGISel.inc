/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* DAG Instruction Selector for the BPF target                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

// *** NOTE: This file is #included into the middle of the target
// *** instruction selector class.  These functions are really methods.

// If GET_DAGISEL_DECL is #defined with any value, only function
// declarations will be included when this file is included.
// If GET_DAGISEL_BODY is #defined, its value should be the name of
// the instruction selector class. Function bodies will be emitted
// and each function's name will be qualified with the name of the
// class.
//
// When neither of the GET_DAGISEL* macros is defined, the functions
// are emitted inline.

#if defined(GET_DAGISEL_DECL) && defined(GET_DAGISEL_BODY)
#error GET_DAGISEL_DECL and GET_DAGISEL_BODY cannot be both defined, undef both for inline definitions
#endif

#ifdef GET_DAGISEL_BODY
#define LOCAL_DAGISEL_STRINGIZE(X) LOCAL_DAGISEL_STRINGIZE_(X)
#define LOCAL_DAGISEL_STRINGIZE_(X) #X
static_assert(sizeof(LOCAL_DAGISEL_STRINGIZE(GET_DAGISEL_BODY)) > 1,
   "GET_DAGISEL_BODY is empty: it should be defined with the class name");
#undef LOCAL_DAGISEL_STRINGIZE_
#undef LOCAL_DAGISEL_STRINGIZE
#endif

#if !defined(GET_DAGISEL_DECL) && !defined(GET_DAGISEL_BODY)
#define DAGISEL_INLINE 1
#else
#define DAGISEL_INLINE 0
#endif

#if !DAGISEL_INLINE
#define DAGISEL_CLASS_COLONCOLON GET_DAGISEL_BODY ::
#else
#define DAGISEL_CLASS_COLONCOLON
#endif

#ifdef GET_DAGISEL_DECL
void SelectCode(SDNode *N);
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
void DAGISEL_CLASS_COLONCOLON SelectCode(SDNode *N)
{
  // Some target values are emitted as 2 bytes, TARGET_VAL handles
  // this.
  #define TARGET_VAL(X) X & 255, unsigned(X) >> 8
  static const unsigned char MatcherTable[] = {
/*     0*/ OPC_SwitchOpcode /*39 cases */, 106|128,2/*362*/, TARGET_VAL(ISD::STORE),// ->367
/*     5*/  OPC_RecordMemRef,
/*     6*/  OPC_RecordNode, // #0 = 'st' chained node
/*     7*/  OPC_RecordChild1, // #1 = $imm
/*     8*/  OPC_Scope, 27|128,1/*155*/, /*->166*/ // 3 children in Scope
/*    11*/   OPC_MoveChild1,
/*    12*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*    15*/   OPC_SwitchType /*2 cases */, 80, MVT::i64,// ->98
/*    18*/    OPC_Scope, 20, /*->40*/ // 2 children in Scope
/*    20*/     OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*    21*/     OPC_MoveParent,
/*    22*/     OPC_RecordChild2, // #2 = $addr
/*    23*/     OPC_CheckChild2TypeI64,
/*    24*/     OPC_CheckPredicate5,  // Predicate_unindexedstore
/*    25*/     OPC_CheckPredicate, 9, // Predicate_store
/*    27*/     OPC_CheckPatternPredicate5, // (Subtarget->hasStoreImm())
/*    28*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*    30*/     OPC_EmitMergeInputChains1_0,
/*    31*/     OPC_EmitConvertToTarget1,
/*    32*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STD_imm), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 5, 3, 4, 
               // Src: (st (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_store>> - Complexity = 17
               // Dst: (STD_imm (imm:{ *:[i64] }):$imm, ADDRri:{ *:[i64] }:$addr)
/*    40*/    /*Scope*/ 56, /*->97*/
/*    41*/     OPC_CheckPredicate, 25, // Predicate_i64immZExt32
/*    43*/     OPC_MoveParent,
/*    44*/     OPC_RecordChild2, // #2 = $addr
/*    45*/     OPC_CheckChild2TypeI64,
/*    46*/     OPC_CheckPredicate5,  // Predicate_unindexedstore
/*    47*/     OPC_CheckPredicate6,  // Predicate_truncstore
/*    48*/     OPC_Scope, 15, /*->65*/ // 3 children in Scope
/*    50*/      OPC_CheckPredicate, 20, // Predicate_truncstorei32
/*    52*/      OPC_CheckPatternPredicate5, // (Subtarget->hasStoreImm())
/*    53*/      OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*    55*/      OPC_EmitMergeInputChains1_0,
/*    56*/      OPC_EmitConvertToTarget1,
/*    57*/      OPC_MorphNodeTo0, TARGET_VAL(BPF::STW_imm), 0|OPFL_Chain|OPFL_MemRefs,
                    3/*#Ops*/, 5, 3, 4, 
                // Src: (st (imm:{ *:[i64] })<<P:Predicate_i64immZExt32>>:$imm, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei32>> - Complexity = 17
                // Dst: (STW_imm (imm:{ *:[i64] }):$imm, ADDRri:{ *:[i64] }:$addr)
/*    65*/     /*Scope*/ 14, /*->80*/
/*    66*/      OPC_CheckPredicate7,  // Predicate_truncstorei16
/*    67*/      OPC_CheckPatternPredicate5, // (Subtarget->hasStoreImm())
/*    68*/      OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*    70*/      OPC_EmitMergeInputChains1_0,
/*    71*/      OPC_EmitConvertToTarget1,
/*    72*/      OPC_MorphNodeTo0, TARGET_VAL(BPF::STH_imm), 0|OPFL_Chain|OPFL_MemRefs,
                    3/*#Ops*/, 5, 3, 4, 
                // Src: (st (imm:{ *:[i64] })<<P:Predicate_i64immZExt32>>:$imm, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei16>> - Complexity = 17
                // Dst: (STH_imm (imm:{ *:[i64] }):$imm, ADDRri:{ *:[i64] }:$addr)
/*    80*/     /*Scope*/ 15, /*->96*/
/*    81*/      OPC_CheckPredicate, 8, // Predicate_truncstorei8
/*    83*/      OPC_CheckPatternPredicate5, // (Subtarget->hasStoreImm())
/*    84*/      OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*    86*/      OPC_EmitMergeInputChains1_0,
/*    87*/      OPC_EmitConvertToTarget1,
/*    88*/      OPC_MorphNodeTo0, TARGET_VAL(BPF::STB_imm), 0|OPFL_Chain|OPFL_MemRefs,
                    3/*#Ops*/, 5, 3, 4, 
                // Src: (st (imm:{ *:[i64] })<<P:Predicate_i64immZExt32>>:$imm, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei8>> - Complexity = 17
                // Dst: (STB_imm (imm:{ *:[i64] }):$imm, ADDRri:{ *:[i64] }:$addr)
/*    96*/     0, /*End of Scope*/
/*    97*/    0, /*End of Scope*/
/*    98*/   /*SwitchType*/ 65, MVT::i32,// ->165
/*   100*/    OPC_MoveParent,
/*   101*/    OPC_RecordChild2, // #2 = $dst
/*   102*/    OPC_CheckChild2TypeI64,
/*   103*/    OPC_CheckPredicate5,  // Predicate_unindexedstore
/*   104*/    OPC_Scope, 17, /*->123*/ // 2 children in Scope
/*   106*/     OPC_CheckPredicate, 9, // Predicate_store
/*   108*/     OPC_CheckPatternPredicate6, // (Subtarget->getHasAlu32()) && (Subtarget->hasStoreImm())
/*   109*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$dst #3 #4
/*   111*/     OPC_EmitMergeInputChains1_0,
/*   112*/     OPC_EmitNodeXForm, 0, 1, // imm_to_i64
/*   115*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STW_imm), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 5, 3, 4, 
               // Src: (st (imm:{ *:[i32] }):$src, ADDRri:{ *:[i64] }:$dst)<<P:Predicate_unindexedstore>><<P:Predicate_store>> - Complexity = 16
               // Dst: (STW_imm (imm_to_i64:{ *:[i64] } ?:{ *:[i32] }:$src), ADDRri:{ *:[i64] }:$dst)
/*   123*/    /*Scope*/ 40, /*->164*/
/*   124*/     OPC_CheckPredicate6,  // Predicate_truncstore
/*   125*/     OPC_Scope, 17, /*->144*/ // 2 children in Scope
/*   127*/      OPC_CheckPredicate7,  // Predicate_truncstorei16
/*   128*/      OPC_CheckPatternPredicate6, // (Subtarget->getHasAlu32()) && (Subtarget->hasStoreImm())
/*   129*/      OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$dst #3 #4
/*   131*/      OPC_EmitMergeInputChains1_0,
/*   132*/      OPC_EmitConvertToTarget1,
/*   133*/      OPC_EmitNodeXForm, 0, 5, // imm_to_i64
/*   136*/      OPC_MorphNodeTo0, TARGET_VAL(BPF::STH_imm), 0|OPFL_Chain|OPFL_MemRefs,
                    3/*#Ops*/, 6, 3, 4, 
                // Src: (st (imm:{ *:[i32] }):$src, ADDRri:{ *:[i64] }:$dst)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei16>> - Complexity = 16
                // Dst: (STH_imm (imm_to_i64:{ *:[i64] } (imm:{ *:[i32] }):$src), ADDRri:{ *:[i64] }:$dst)
/*   144*/     /*Scope*/ 18, /*->163*/
/*   145*/      OPC_CheckPredicate, 8, // Predicate_truncstorei8
/*   147*/      OPC_CheckPatternPredicate6, // (Subtarget->getHasAlu32()) && (Subtarget->hasStoreImm())
/*   148*/      OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$dst #3 #4
/*   150*/      OPC_EmitMergeInputChains1_0,
/*   151*/      OPC_EmitConvertToTarget1,
/*   152*/      OPC_EmitNodeXForm, 0, 5, // imm_to_i64
/*   155*/      OPC_MorphNodeTo0, TARGET_VAL(BPF::STB_imm), 0|OPFL_Chain|OPFL_MemRefs,
                    3/*#Ops*/, 6, 3, 4, 
                // Src: (st (imm:{ *:[i32] }):$src, ADDRri:{ *:[i64] }:$dst)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei8>> - Complexity = 16
                // Dst: (STB_imm (imm_to_i64:{ *:[i64] } (imm:{ *:[i32] }):$src), ADDRri:{ *:[i64] }:$dst)
/*   163*/     0, /*End of Scope*/
/*   164*/    0, /*End of Scope*/
/*   165*/   0, // EndSwitchType
/*   166*/  /*Scope*/ 15|128,1/*143*/, /*->311*/
/*   168*/   OPC_CheckChild1TypeI64,
/*   169*/   OPC_RecordChild2, // #2 = $addr
/*   170*/   OPC_CheckChild2TypeI64,
/*   171*/   OPC_CheckPredicate5,  // Predicate_unindexedstore
/*   172*/   OPC_Scope, 47, /*->221*/ // 3 children in Scope
/*   174*/    OPC_CheckPredicate6,  // Predicate_truncstore
/*   175*/    OPC_Scope, 14, /*->191*/ // 3 children in Scope
/*   177*/     OPC_CheckPredicate, 20, // Predicate_truncstorei32
/*   179*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   180*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*   182*/     OPC_EmitMergeInputChains1_0,
/*   183*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STW), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 1, 3, 4, 
               // Src: (st GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei32>> - Complexity = 13
               // Dst: (STW GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$addr)
/*   191*/    /*Scope*/ 13, /*->205*/
/*   192*/     OPC_CheckPredicate7,  // Predicate_truncstorei16
/*   193*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   194*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*   196*/     OPC_EmitMergeInputChains1_0,
/*   197*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STH), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 1, 3, 4, 
               // Src: (st GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei16>> - Complexity = 13
               // Dst: (STH GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$addr)
/*   205*/    /*Scope*/ 14, /*->220*/
/*   206*/     OPC_CheckPredicate, 8, // Predicate_truncstorei8
/*   208*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   209*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*   211*/     OPC_EmitMergeInputChains1_0,
/*   212*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STB), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 1, 3, 4, 
               // Src: (st GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei8>> - Complexity = 13
               // Dst: (STB GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$addr)
/*   220*/    0, /*End of Scope*/
/*   221*/   /*Scope*/ 13, /*->235*/
/*   222*/    OPC_CheckPredicate, 9, // Predicate_store
/*   224*/    OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*   226*/    OPC_EmitMergeInputChains1_0,
/*   227*/    OPC_MorphNodeTo0, TARGET_VAL(BPF::STD), 0|OPFL_Chain|OPFL_MemRefs,
                  3/*#Ops*/, 1, 3, 4, 
              // Src: (st GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_store>> - Complexity = 13
              // Dst: (STD GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$addr)
/*   235*/   /*Scope*/ 74, /*->310*/
/*   236*/    OPC_CheckPredicate6,  // Predicate_truncstore
/*   237*/    OPC_Scope, 23, /*->262*/ // 3 children in Scope
/*   239*/     OPC_CheckPredicate, 8, // Predicate_truncstorei8
/*   241*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   242*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$dst #3 #4
/*   244*/     OPC_EmitMergeInputChains1_0,
/*   245*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   247*/     OPC_EmitNode1None, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG),
                   MVT::i32, 2/*#Ops*/, 1, 5,  // Results = #6
/*   254*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STB32), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 6, 3, 4, 
               // Src: (st GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$dst)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei8>> - Complexity = 13
               // Dst: (STB32 (EXTRACT_SUBREG:{ *:[i32] } GPR:{ *:[i64] }:$src, sub_32:{ *:[i32] }), ADDRri:{ *:[i64] }:$dst)
/*   262*/    /*Scope*/ 22, /*->285*/
/*   263*/     OPC_CheckPredicate7,  // Predicate_truncstorei16
/*   264*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   265*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$dst #3 #4
/*   267*/     OPC_EmitMergeInputChains1_0,
/*   268*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   270*/     OPC_EmitNode1None, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG),
                   MVT::i32, 2/*#Ops*/, 1, 5,  // Results = #6
/*   277*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STH32), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 6, 3, 4, 
               // Src: (st GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$dst)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei16>> - Complexity = 13
               // Dst: (STH32 (EXTRACT_SUBREG:{ *:[i32] } GPR:{ *:[i64] }:$src, sub_32:{ *:[i32] }), ADDRri:{ *:[i64] }:$dst)
/*   285*/    /*Scope*/ 23, /*->309*/
/*   286*/     OPC_CheckPredicate, 20, // Predicate_truncstorei32
/*   288*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   289*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$dst #3 #4
/*   291*/     OPC_EmitMergeInputChains1_0,
/*   292*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   294*/     OPC_EmitNode1None, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG),
                   MVT::i32, 2/*#Ops*/, 1, 5,  // Results = #6
/*   301*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STW32), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 6, 3, 4, 
               // Src: (st GPR:{ *:[i64] }:$src, ADDRri:{ *:[i64] }:$dst)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei32>> - Complexity = 13
               // Dst: (STW32 (EXTRACT_SUBREG:{ *:[i32] } GPR:{ *:[i64] }:$src, sub_32:{ *:[i32] }), ADDRri:{ *:[i64] }:$dst)
/*   309*/    0, /*End of Scope*/
/*   310*/   0, /*End of Scope*/
/*   311*/  /*Scope*/ 54, /*->366*/
/*   312*/   OPC_CheckChild1TypeI32,
/*   313*/   OPC_RecordChild2, // #2 = $addr
/*   314*/   OPC_CheckChild2TypeI64,
/*   315*/   OPC_CheckPredicate5,  // Predicate_unindexedstore
/*   316*/   OPC_Scope, 14, /*->332*/ // 2 children in Scope
/*   318*/    OPC_CheckPredicate, 9, // Predicate_store
/*   320*/    OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   321*/    OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*   323*/    OPC_EmitMergeInputChains1_0,
/*   324*/    OPC_MorphNodeTo0, TARGET_VAL(BPF::STW32), 0|OPFL_Chain|OPFL_MemRefs,
                  3/*#Ops*/, 1, 3, 4, 
              // Src: (st GPR32:{ *:[i32] }:$src, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_store>> - Complexity = 13
              // Dst: (STW32 GPR32:{ *:[i32] }:$src, ADDRri:{ *:[i64] }:$addr)
/*   332*/   /*Scope*/ 32, /*->365*/
/*   333*/    OPC_CheckPredicate6,  // Predicate_truncstore
/*   334*/    OPC_Scope, 13, /*->349*/ // 2 children in Scope
/*   336*/     OPC_CheckPredicate7,  // Predicate_truncstorei16
/*   337*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   338*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*   340*/     OPC_EmitMergeInputChains1_0,
/*   341*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STH32), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 1, 3, 4, 
               // Src: (st GPR32:{ *:[i32] }:$src, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei16>> - Complexity = 13
               // Dst: (STH32 GPR32:{ *:[i32] }:$src, ADDRri:{ *:[i64] }:$addr)
/*   349*/    /*Scope*/ 14, /*->364*/
/*   350*/     OPC_CheckPredicate, 8, // Predicate_truncstorei8
/*   352*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   353*/     OPC_CheckComplexPat0, /*#*/2, // SelectAddr:$addr #3 #4
/*   355*/     OPC_EmitMergeInputChains1_0,
/*   356*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::STB32), 0|OPFL_Chain|OPFL_MemRefs,
                   3/*#Ops*/, 1, 3, 4, 
               // Src: (st GPR32:{ *:[i32] }:$src, ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei8>> - Complexity = 13
               // Dst: (STB32 GPR32:{ *:[i32] }:$src, ADDRri:{ *:[i64] }:$addr)
/*   364*/    0, /*End of Scope*/
/*   365*/   0, /*End of Scope*/
/*   366*/  0, /*End of Scope*/
/*   367*/ /*SwitchOpcode*/ 2|128,1/*130*/, TARGET_VAL(ISD::INTRINSIC_W_CHAIN),// ->501
/*   371*/  OPC_RecordNode, // #0 = 'intrinsic_w_chain' chained node
/*   372*/  OPC_Scope, 24, /*->398*/ // 4 children in Scope
/*   374*/   OPC_CheckChild1Integer, 14|128,52/*6670*/, 
/*   377*/   OPC_RecordChild2, // #1 = $pseudo
/*   378*/   OPC_MoveChild2,
/*   379*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   382*/   OPC_MoveSibling3,
/*   383*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   386*/   OPC_RecordNode, // #2 = $imm
/*   387*/   OPC_MoveParent,
/*   388*/   OPC_EmitMergeInputChains1_0,
/*   389*/   OPC_EmitConvertToTarget1,
/*   390*/   OPC_EmitConvertToTarget2,
/*   391*/   OPC_MorphNodeTo1Chain, TARGET_VAL(BPF::LD_pseudo),
                 MVT::i64, 2/*#Ops*/, 3, 4, 
             // Src: (intrinsic_w_chain:{ *:[i64] } 3335:{ *:[iPTR] }, (imm:{ *:[i64] }):$pseudo, (imm:{ *:[i64] }):$imm) - Complexity = 14
             // Dst: (LD_pseudo:{ *:[i64] } (imm:{ *:[i64] }):$pseudo, (imm:{ *:[i64] }):$imm)
/*   398*/  /*Scope*/ 33, /*->432*/
/*   399*/   OPC_CheckChild1Integer, 0|128,52/*6656*/, 
/*   402*/   OPC_RecordChild2, // #1 = $skb
/*   403*/   OPC_CheckChild2TypeI64,
/*   404*/   OPC_RecordChild3, // #2 = $imm
/*   405*/   OPC_Scope, 15, /*->422*/ // 2 children in Scope
/*   407*/    OPC_MoveChild3,
/*   408*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   411*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*   412*/    OPC_MoveParent,
/*   413*/    OPC_EmitMergeInputChains1_0,
/*   414*/    OPC_EmitConvertToTarget2,
/*   415*/    OPC_MorphNodeTo1Chain, TARGET_VAL(BPF::LD_ABS_B),
                  MVT::i64, 2/*#Ops*/, 1, 3, 
              // Src: (intrinsic_w_chain:{ *:[i64] } 3328:{ *:[iPTR] }, GPR:{ *:[i64] }:$skb, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 12
              // Dst: (LD_ABS_B:{ *:[i64] } GPR:{ *:[i64] }:$skb, (imm:{ *:[i64] }):$imm)
/*   422*/   /*Scope*/ 8, /*->431*/
/*   423*/    OPC_EmitMergeInputChains1_0,
/*   424*/    OPC_MorphNodeTo1Chain, TARGET_VAL(BPF::LD_IND_B),
                  MVT::i64, 2/*#Ops*/, 1, 2, 
              // Src: (intrinsic_w_chain:{ *:[i64] } 3328:{ *:[iPTR] }, GPR:{ *:[i64] }:$skb, GPR:{ *:[i64] }:$val) - Complexity = 8
              // Dst: (LD_IND_B:{ *:[i64] } GPR:{ *:[i64] }:$skb, GPR:{ *:[i64] }:$val)
/*   431*/   0, /*End of Scope*/
/*   432*/  /*Scope*/ 33, /*->466*/
/*   433*/   OPC_CheckChild1Integer, 2|128,52/*6658*/, 
/*   436*/   OPC_RecordChild2, // #1 = $skb
/*   437*/   OPC_CheckChild2TypeI64,
/*   438*/   OPC_RecordChild3, // #2 = $imm
/*   439*/   OPC_Scope, 15, /*->456*/ // 2 children in Scope
/*   441*/    OPC_MoveChild3,
/*   442*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   445*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*   446*/    OPC_MoveParent,
/*   447*/    OPC_EmitMergeInputChains1_0,
/*   448*/    OPC_EmitConvertToTarget2,
/*   449*/    OPC_MorphNodeTo1Chain, TARGET_VAL(BPF::LD_ABS_H),
                  MVT::i64, 2/*#Ops*/, 1, 3, 
              // Src: (intrinsic_w_chain:{ *:[i64] } 3329:{ *:[iPTR] }, GPR:{ *:[i64] }:$skb, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 12
              // Dst: (LD_ABS_H:{ *:[i64] } GPR:{ *:[i64] }:$skb, (imm:{ *:[i64] }):$imm)
/*   456*/   /*Scope*/ 8, /*->465*/
/*   457*/    OPC_EmitMergeInputChains1_0,
/*   458*/    OPC_MorphNodeTo1Chain, TARGET_VAL(BPF::LD_IND_H),
                  MVT::i64, 2/*#Ops*/, 1, 2, 
              // Src: (intrinsic_w_chain:{ *:[i64] } 3329:{ *:[iPTR] }, GPR:{ *:[i64] }:$skb, GPR:{ *:[i64] }:$val) - Complexity = 8
              // Dst: (LD_IND_H:{ *:[i64] } GPR:{ *:[i64] }:$skb, GPR:{ *:[i64] }:$val)
/*   465*/   0, /*End of Scope*/
/*   466*/  /*Scope*/ 33, /*->500*/
/*   467*/   OPC_CheckChild1Integer, 4|128,52/*6660*/, 
/*   470*/   OPC_RecordChild2, // #1 = $skb
/*   471*/   OPC_CheckChild2TypeI64,
/*   472*/   OPC_RecordChild3, // #2 = $imm
/*   473*/   OPC_Scope, 15, /*->490*/ // 2 children in Scope
/*   475*/    OPC_MoveChild3,
/*   476*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   479*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*   480*/    OPC_MoveParent,
/*   481*/    OPC_EmitMergeInputChains1_0,
/*   482*/    OPC_EmitConvertToTarget2,
/*   483*/    OPC_MorphNodeTo1Chain, TARGET_VAL(BPF::LD_ABS_W),
                  MVT::i64, 2/*#Ops*/, 1, 3, 
              // Src: (intrinsic_w_chain:{ *:[i64] } 3330:{ *:[iPTR] }, GPR:{ *:[i64] }:$skb, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 12
              // Dst: (LD_ABS_W:{ *:[i64] } GPR:{ *:[i64] }:$skb, (imm:{ *:[i64] }):$imm)
/*   490*/   /*Scope*/ 8, /*->499*/
/*   491*/    OPC_EmitMergeInputChains1_0,
/*   492*/    OPC_MorphNodeTo1Chain, TARGET_VAL(BPF::LD_IND_W),
                  MVT::i64, 2/*#Ops*/, 1, 2, 
              // Src: (intrinsic_w_chain:{ *:[i64] } 3330:{ *:[iPTR] }, GPR:{ *:[i64] }:$skb, GPR:{ *:[i64] }:$val) - Complexity = 8
              // Dst: (LD_IND_W:{ *:[i64] } GPR:{ *:[i64] }:$skb, GPR:{ *:[i64] }:$val)
/*   499*/   0, /*End of Scope*/
/*   500*/  0, /*End of Scope*/
/*   501*/ /*SwitchOpcode*/ 87|128,3/*471*/, TARGET_VAL(ISD::LOAD),// ->976
/*   505*/  OPC_RecordMemRef,
/*   506*/  OPC_RecordNode, // #0 = 'ld' chained node
/*   507*/  OPC_RecordChild1, // #1 = $addr
/*   508*/  OPC_CheckChild1TypeI64,
/*   509*/  OPC_CheckPredicate, 26, // Predicate_unindexedload
/*   511*/  OPC_SwitchType /*2 cases */, 67|128,2/*323*/, MVT::i64,// ->838
/*   515*/   OPC_Scope, 46, /*->563*/ // 6 children in Scope
/*   517*/    OPC_CheckPredicate, 21, // Predicate_zextload
/*   519*/    OPC_Scope, 13, /*->534*/ // 3 children in Scope
/*   521*/     OPC_CheckPredicate1,  // Predicate_zextloadi32
/*   522*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   523*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   525*/     OPC_EmitMergeInputChains1_0,
/*   526*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDW), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi32>> - Complexity = 13
               // Dst: (LDW:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)
/*   534*/    /*Scope*/ 13, /*->548*/
/*   535*/     OPC_CheckPredicate2,  // Predicate_zextloadi16
/*   536*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   537*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   539*/     OPC_EmitMergeInputChains1_0,
/*   540*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDH), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi16>> - Complexity = 13
               // Dst: (LDH:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)
/*   548*/    /*Scope*/ 13, /*->562*/
/*   549*/     OPC_CheckPredicate3,  // Predicate_zextloadi8
/*   550*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   551*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   553*/     OPC_EmitMergeInputChains1_0,
/*   554*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDB), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi8>> - Complexity = 13
               // Dst: (LDB:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)
/*   562*/    0, /*End of Scope*/
/*   563*/   /*Scope*/ 46, /*->610*/
/*   564*/    OPC_CheckPredicate, 23, // Predicate_sextload
/*   566*/    OPC_Scope, 13, /*->581*/ // 3 children in Scope
/*   568*/     OPC_CheckPredicate1,  // Predicate_sextloadi32
/*   569*/     OPC_CheckPatternPredicate4, // (Subtarget->hasLdsx())
/*   570*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   572*/     OPC_EmitMergeInputChains1_0,
/*   573*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDWSX), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi32>> - Complexity = 13
               // Dst: (LDWSX:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)
/*   581*/    /*Scope*/ 13, /*->595*/
/*   582*/     OPC_CheckPredicate2,  // Predicate_sextloadi16
/*   583*/     OPC_CheckPatternPredicate4, // (Subtarget->hasLdsx())
/*   584*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   586*/     OPC_EmitMergeInputChains1_0,
/*   587*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDHSX), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi16>> - Complexity = 13
               // Dst: (LDHSX:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)
/*   595*/    /*Scope*/ 13, /*->609*/
/*   596*/     OPC_CheckPredicate3,  // Predicate_sextloadi8
/*   597*/     OPC_CheckPatternPredicate4, // (Subtarget->hasLdsx())
/*   598*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   600*/     OPC_EmitMergeInputChains1_0,
/*   601*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDBSX), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi8>> - Complexity = 13
               // Dst: (LDBSX:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)
/*   609*/    0, /*End of Scope*/
/*   610*/   /*Scope*/ 13, /*->624*/
/*   611*/    OPC_CheckPredicate, 24, // Predicate_load
/*   613*/    OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   615*/    OPC_EmitMergeInputChains1_0,
/*   616*/    OPC_MorphNodeTo1, TARGET_VAL(BPF::LDD), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i64, 2/*#Ops*/, 2, 3, 
              // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_load>> - Complexity = 13
              // Dst: (LDD:{ *:[i64] } ADDRri:{ *:[i64] }:$addr)
/*   624*/   /*Scope*/ 46, /*->671*/
/*   625*/    OPC_CheckPredicate, 22, // Predicate_extload
/*   627*/    OPC_Scope, 13, /*->642*/ // 3 children in Scope
/*   629*/     OPC_CheckPredicate3,  // Predicate_extloadi8
/*   630*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   631*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   633*/     OPC_EmitMergeInputChains1_0,
/*   634*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDB), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi8>> - Complexity = 13
               // Dst: (LDB:{ *:[i64] } ADDRri:{ *:[i64] }:$src)
/*   642*/    /*Scope*/ 13, /*->656*/
/*   643*/     OPC_CheckPredicate2,  // Predicate_extloadi16
/*   644*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   645*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   647*/     OPC_EmitMergeInputChains1_0,
/*   648*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDH), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi16>> - Complexity = 13
               // Dst: (LDH:{ *:[i64] } ADDRri:{ *:[i64] }:$src)
/*   656*/    /*Scope*/ 13, /*->670*/
/*   657*/     OPC_CheckPredicate1,  // Predicate_extloadi32
/*   658*/     OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   659*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   661*/     OPC_EmitMergeInputChains1_0,
/*   662*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDW), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi32>> - Complexity = 13
               // Dst: (LDW:{ *:[i64] } ADDRri:{ *:[i64] }:$src)
/*   670*/    0, /*End of Scope*/
/*   671*/   /*Scope*/ 82, /*->754*/
/*   672*/    OPC_CheckPredicate, 21, // Predicate_zextload
/*   674*/    OPC_Scope, 25, /*->701*/ // 3 children in Scope
/*   676*/     OPC_CheckPredicate3,  // Predicate_zextloadi8
/*   677*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   678*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   680*/     OPC_EmitMergeInputChains1_0,
/*   681*/     OPC_EmitInteger64, 0, 
/*   683*/     OPC_EmitNode1, TARGET_VAL(BPF::LDB32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #5
/*   691*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   693*/     OPC_MorphNodeTo1Chain, TARGET_VAL(TargetOpcode::SUBREG_TO_REG),
                   MVT::i64, 3/*#Ops*/, 4, 5, 6, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi8>> - Complexity = 13
               // Dst: (SUBREG_TO_REG:{ *:[i64] } 0:{ *:[i64] }, (LDB32:{ *:[i32] } ADDRri:{ *:[i64] }:$src), sub_32:{ *:[i32] })
/*   701*/    /*Scope*/ 25, /*->727*/
/*   702*/     OPC_CheckPredicate2,  // Predicate_zextloadi16
/*   703*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   704*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   706*/     OPC_EmitMergeInputChains1_0,
/*   707*/     OPC_EmitInteger64, 0, 
/*   709*/     OPC_EmitNode1, TARGET_VAL(BPF::LDH32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #5
/*   717*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   719*/     OPC_MorphNodeTo1Chain, TARGET_VAL(TargetOpcode::SUBREG_TO_REG),
                   MVT::i64, 3/*#Ops*/, 4, 5, 6, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi16>> - Complexity = 13
               // Dst: (SUBREG_TO_REG:{ *:[i64] } 0:{ *:[i64] }, (LDH32:{ *:[i32] } ADDRri:{ *:[i64] }:$src), sub_32:{ *:[i32] })
/*   727*/    /*Scope*/ 25, /*->753*/
/*   728*/     OPC_CheckPredicate1,  // Predicate_zextloadi32
/*   729*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   730*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   732*/     OPC_EmitMergeInputChains1_0,
/*   733*/     OPC_EmitInteger64, 0, 
/*   735*/     OPC_EmitNode1, TARGET_VAL(BPF::LDW32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #5
/*   743*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   745*/     OPC_MorphNodeTo1Chain, TARGET_VAL(TargetOpcode::SUBREG_TO_REG),
                   MVT::i64, 3/*#Ops*/, 4, 5, 6, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi32>> - Complexity = 13
               // Dst: (SUBREG_TO_REG:{ *:[i64] } 0:{ *:[i64] }, (LDW32:{ *:[i32] } ADDRri:{ *:[i64] }:$src), sub_32:{ *:[i32] })
/*   753*/    0, /*End of Scope*/
/*   754*/   /*Scope*/ 82, /*->837*/
/*   755*/    OPC_CheckPredicate, 22, // Predicate_extload
/*   757*/    OPC_Scope, 25, /*->784*/ // 3 children in Scope
/*   759*/     OPC_CheckPredicate3,  // Predicate_extloadi8
/*   760*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   761*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   763*/     OPC_EmitMergeInputChains1_0,
/*   764*/     OPC_EmitInteger64, 0, 
/*   766*/     OPC_EmitNode1, TARGET_VAL(BPF::LDB32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #5
/*   774*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   776*/     OPC_MorphNodeTo1Chain, TARGET_VAL(TargetOpcode::SUBREG_TO_REG),
                   MVT::i64, 3/*#Ops*/, 4, 5, 6, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi8>> - Complexity = 13
               // Dst: (SUBREG_TO_REG:{ *:[i64] } 0:{ *:[i64] }, (LDB32:{ *:[i32] } ADDRri:{ *:[i64] }:$src), sub_32:{ *:[i32] })
/*   784*/    /*Scope*/ 25, /*->810*/
/*   785*/     OPC_CheckPredicate2,  // Predicate_extloadi16
/*   786*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   787*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   789*/     OPC_EmitMergeInputChains1_0,
/*   790*/     OPC_EmitInteger64, 0, 
/*   792*/     OPC_EmitNode1, TARGET_VAL(BPF::LDH32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #5
/*   800*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   802*/     OPC_MorphNodeTo1Chain, TARGET_VAL(TargetOpcode::SUBREG_TO_REG),
                   MVT::i64, 3/*#Ops*/, 4, 5, 6, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi16>> - Complexity = 13
               // Dst: (SUBREG_TO_REG:{ *:[i64] } 0:{ *:[i64] }, (LDH32:{ *:[i32] } ADDRri:{ *:[i64] }:$src), sub_32:{ *:[i32] })
/*   810*/    /*Scope*/ 25, /*->836*/
/*   811*/     OPC_CheckPredicate1,  // Predicate_extloadi32
/*   812*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   813*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   815*/     OPC_EmitMergeInputChains1_0,
/*   816*/     OPC_EmitInteger64, 0, 
/*   818*/     OPC_EmitNode1, TARGET_VAL(BPF::LDW32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #5
/*   826*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   828*/     OPC_MorphNodeTo1Chain, TARGET_VAL(TargetOpcode::SUBREG_TO_REG),
                   MVT::i64, 3/*#Ops*/, 4, 5, 6, 
               // Src: (ld:{ *:[i64] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi32>> - Complexity = 13
               // Dst: (SUBREG_TO_REG:{ *:[i64] } 0:{ *:[i64] }, (LDW32:{ *:[i32] } ADDRri:{ *:[i64] }:$src), sub_32:{ *:[i32] })
/*   836*/    0, /*End of Scope*/
/*   837*/   0, /*End of Scope*/
/*   838*/  /*SwitchType*/ 6|128,1/*134*/, MVT::i32,// ->975
/*   841*/   OPC_Scope, 14, /*->857*/ // 4 children in Scope
/*   843*/    OPC_CheckPredicate, 24, // Predicate_load
/*   845*/    OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   846*/    OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   848*/    OPC_EmitMergeInputChains1_0,
/*   849*/    OPC_MorphNodeTo1, TARGET_VAL(BPF::LDW32), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (ld:{ *:[i32] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_load>> - Complexity = 13
              // Dst: (LDW32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr)
/*   857*/   /*Scope*/ 32, /*->890*/
/*   858*/    OPC_CheckPredicate, 21, // Predicate_zextload
/*   860*/    OPC_Scope, 13, /*->875*/ // 2 children in Scope
/*   862*/     OPC_CheckPredicate2,  // Predicate_zextloadi16
/*   863*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   864*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   866*/     OPC_EmitMergeInputChains1_0,
/*   867*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDH32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i32] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi16>> - Complexity = 13
               // Dst: (LDH32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr)
/*   875*/    /*Scope*/ 13, /*->889*/
/*   876*/     OPC_CheckPredicate3,  // Predicate_zextloadi8
/*   877*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   878*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #2 #3
/*   880*/     OPC_EmitMergeInputChains1_0,
/*   881*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDB32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i32] } ADDRri:{ *:[i64] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi8>> - Complexity = 13
               // Dst: (LDB32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr)
/*   889*/    0, /*End of Scope*/
/*   890*/   /*Scope*/ 32, /*->923*/
/*   891*/    OPC_CheckPredicate, 22, // Predicate_extload
/*   893*/    OPC_Scope, 13, /*->908*/ // 2 children in Scope
/*   895*/     OPC_CheckPredicate3,  // Predicate_extloadi8
/*   896*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   897*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   899*/     OPC_EmitMergeInputChains1_0,
/*   900*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDB32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i32] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi8>> - Complexity = 13
               // Dst: (LDB32:{ *:[i32] } ADDRri:{ *:[i64] }:$src)
/*   908*/    /*Scope*/ 13, /*->922*/
/*   909*/     OPC_CheckPredicate2,  // Predicate_extloadi16
/*   910*/     OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*   911*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   913*/     OPC_EmitMergeInputChains1_0,
/*   914*/     OPC_MorphNodeTo1, TARGET_VAL(BPF::LDH32), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i32, 2/*#Ops*/, 2, 3, 
               // Src: (ld:{ *:[i32] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi16>> - Complexity = 13
               // Dst: (LDH32:{ *:[i32] } ADDRri:{ *:[i64] }:$src)
/*   922*/    0, /*End of Scope*/
/*   923*/   /*Scope*/ 50, /*->974*/
/*   924*/    OPC_CheckPredicate, 23, // Predicate_sextload
/*   926*/    OPC_Scope, 22, /*->950*/ // 2 children in Scope
/*   928*/     OPC_CheckPredicate3,  // Predicate_sextloadi8
/*   929*/     OPC_CheckPatternPredicate4, // (Subtarget->hasLdsx())
/*   930*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   932*/     OPC_EmitMergeInputChains1_0,
/*   933*/     OPC_EmitNode1, TARGET_VAL(BPF::LDBSX), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3,  // Results = #4
/*   941*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   943*/     OPC_MorphNodeTo1Chain, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG),
                   MVT::i32, 2/*#Ops*/, 4, 5, 
               // Src: (ld:{ *:[i32] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi8>> - Complexity = 13
               // Dst: (EXTRACT_SUBREG:{ *:[i32] } (LDBSX:{ *:[i64] } ADDRri:{ *:[i64] }:$src), sub_32:{ *:[i32] })
/*   950*/    /*Scope*/ 22, /*->973*/
/*   951*/     OPC_CheckPredicate2,  // Predicate_sextloadi16
/*   952*/     OPC_CheckPatternPredicate4, // (Subtarget->hasLdsx())
/*   953*/     OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$src #2 #3
/*   955*/     OPC_EmitMergeInputChains1_0,
/*   956*/     OPC_EmitNode1, TARGET_VAL(BPF::LDHSX), 0|OPFL_Chain|OPFL_MemRefs,
                   MVT::i64, 2/*#Ops*/, 2, 3,  // Results = #4
/*   964*/     OPC_EmitStringInteger32, BPF::sub_32,
/*   966*/     OPC_MorphNodeTo1Chain, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG),
                   MVT::i32, 2/*#Ops*/, 4, 5, 
               // Src: (ld:{ *:[i32] } ADDRri:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi16>> - Complexity = 13
               // Dst: (EXTRACT_SUBREG:{ *:[i32] } (LDHSX:{ *:[i64] } ADDRri:{ *:[i64] }:$src), sub_32:{ *:[i32] })
/*   973*/    0, /*End of Scope*/
/*   974*/   0, /*End of Scope*/
/*   975*/  0, // EndSwitchType
/*   976*/ /*SwitchOpcode*/ 56, TARGET_VAL(ISD::ATOMIC_LOAD_ADD),// ->1035
/*   979*/  OPC_RecordMemRef,
/*   980*/  OPC_RecordNode, // #0 = 'atomic_load_add' chained node
/*   981*/  OPC_RecordChild1, // #1 = $addr
/*   982*/  OPC_CheckChild1TypeI64,
/*   983*/  OPC_RecordChild2, // #2 = $val
/*   984*/  OPC_Scope, 33, /*->1019*/ // 2 children in Scope
/*   986*/   OPC_CheckPredicate1,  // Predicate_atomic_load_add_32
/*   987*/   OPC_SwitchType /*2 cases */, 13, MVT::i64,// ->1003
/*   990*/    OPC_CheckPatternPredicate1, // (!Subtarget->getHasAlu32())
/*   991*/    OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*   993*/    OPC_EmitMergeInputChains1_0,
/*   994*/    OPC_MorphNodeTo1, TARGET_VAL(BPF::XADDW), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i64, 3/*#Ops*/, 3, 4, 2, 
              // Src: (atomic_load_add:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)<<P:Predicate_atomic_load_add_32>> - Complexity = 13
              // Dst: (XADDW:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)
/*  1003*/   /*SwitchType*/ 13, MVT::i32,// ->1018
/*  1005*/    OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*  1006*/    OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1008*/    OPC_EmitMergeInputChains1_0,
/*  1009*/    OPC_MorphNodeTo1, TARGET_VAL(BPF::XFADDW32), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 3/*#Ops*/, 3, 4, 2, 
              // Src: (atomic_load_add:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)<<P:Predicate_atomic_load_add_32>> - Complexity = 13
              // Dst: (XFADDW32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)
/*  1018*/   0, // EndSwitchType
/*  1019*/  /*Scope*/ 14, /*->1034*/
/*  1020*/   OPC_CheckPredicate4,  // Predicate_atomic_load_add_64
/*  1021*/   OPC_CheckTypeI64,
/*  1022*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1024*/   OPC_EmitMergeInputChains1_0,
/*  1025*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFADDD), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i64, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_load_add:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)<<P:Predicate_atomic_load_add_64>> - Complexity = 13
             // Dst: (XFADDD:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)
/*  1034*/  0, /*End of Scope*/
/*  1035*/ /*SwitchOpcode*/ 38, TARGET_VAL(ISD::ATOMIC_LOAD_AND),// ->1076
/*  1038*/  OPC_RecordMemRef,
/*  1039*/  OPC_RecordNode, // #0 = 'atomic_load_and' chained node
/*  1040*/  OPC_RecordChild1, // #1 = $addr
/*  1041*/  OPC_CheckChild1TypeI64,
/*  1042*/  OPC_RecordChild2, // #2 = $val
/*  1043*/  OPC_SwitchType /*2 cases */, 14, MVT::i32,// ->1060
/*  1046*/   OPC_CheckPredicate1,  // Predicate_atomic_load_and_32
/*  1047*/   OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*  1048*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1050*/   OPC_EmitMergeInputChains1_0,
/*  1051*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFANDW32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_load_and:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)<<P:Predicate_atomic_load_and_32>> - Complexity = 13
             // Dst: (XFANDW32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)
/*  1060*/  /*SwitchType*/ 13, MVT::i64,// ->1075
/*  1062*/   OPC_CheckPredicate4,  // Predicate_atomic_load_and_64
/*  1063*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1065*/   OPC_EmitMergeInputChains1_0,
/*  1066*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFANDD), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i64, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_load_and:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)<<P:Predicate_atomic_load_and_64>> - Complexity = 13
             // Dst: (XFANDD:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)
/*  1075*/  0, // EndSwitchType
/*  1076*/ /*SwitchOpcode*/ 38, TARGET_VAL(ISD::ATOMIC_LOAD_OR),// ->1117
/*  1079*/  OPC_RecordMemRef,
/*  1080*/  OPC_RecordNode, // #0 = 'atomic_load_or' chained node
/*  1081*/  OPC_RecordChild1, // #1 = $addr
/*  1082*/  OPC_CheckChild1TypeI64,
/*  1083*/  OPC_RecordChild2, // #2 = $val
/*  1084*/  OPC_SwitchType /*2 cases */, 14, MVT::i32,// ->1101
/*  1087*/   OPC_CheckPredicate1,  // Predicate_atomic_load_or_32
/*  1088*/   OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*  1089*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1091*/   OPC_EmitMergeInputChains1_0,
/*  1092*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFORW32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_load_or:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)<<P:Predicate_atomic_load_or_32>> - Complexity = 13
             // Dst: (XFORW32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)
/*  1101*/  /*SwitchType*/ 13, MVT::i64,// ->1116
/*  1103*/   OPC_CheckPredicate4,  // Predicate_atomic_load_or_64
/*  1104*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1106*/   OPC_EmitMergeInputChains1_0,
/*  1107*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFORD), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i64, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_load_or:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)<<P:Predicate_atomic_load_or_64>> - Complexity = 13
             // Dst: (XFORD:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)
/*  1116*/  0, // EndSwitchType
/*  1117*/ /*SwitchOpcode*/ 38, TARGET_VAL(ISD::ATOMIC_LOAD_XOR),// ->1158
/*  1120*/  OPC_RecordMemRef,
/*  1121*/  OPC_RecordNode, // #0 = 'atomic_load_xor' chained node
/*  1122*/  OPC_RecordChild1, // #1 = $addr
/*  1123*/  OPC_CheckChild1TypeI64,
/*  1124*/  OPC_RecordChild2, // #2 = $val
/*  1125*/  OPC_SwitchType /*2 cases */, 14, MVT::i32,// ->1142
/*  1128*/   OPC_CheckPredicate1,  // Predicate_atomic_load_xor_32
/*  1129*/   OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*  1130*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1132*/   OPC_EmitMergeInputChains1_0,
/*  1133*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFXORW32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_load_xor:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)<<P:Predicate_atomic_load_xor_32>> - Complexity = 13
             // Dst: (XFXORW32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)
/*  1142*/  /*SwitchType*/ 13, MVT::i64,// ->1157
/*  1144*/   OPC_CheckPredicate4,  // Predicate_atomic_load_xor_64
/*  1145*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1147*/   OPC_EmitMergeInputChains1_0,
/*  1148*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFXORD), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i64, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_load_xor:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)<<P:Predicate_atomic_load_xor_64>> - Complexity = 13
             // Dst: (XFXORD:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)
/*  1157*/  0, // EndSwitchType
/*  1158*/ /*SwitchOpcode*/ 38, TARGET_VAL(ISD::ATOMIC_SWAP),// ->1199
/*  1161*/  OPC_RecordMemRef,
/*  1162*/  OPC_RecordNode, // #0 = 'atomic_swap' chained node
/*  1163*/  OPC_RecordChild1, // #1 = $addr
/*  1164*/  OPC_CheckChild1TypeI64,
/*  1165*/  OPC_RecordChild2, // #2 = $val
/*  1166*/  OPC_SwitchType /*2 cases */, 14, MVT::i32,// ->1183
/*  1169*/   OPC_CheckPredicate1,  // Predicate_atomic_swap_32
/*  1170*/   OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*  1171*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1173*/   OPC_EmitMergeInputChains1_0,
/*  1174*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XCHGW32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_swap:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)<<P:Predicate_atomic_swap_32>> - Complexity = 13
             // Dst: (XCHGW32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)
/*  1183*/  /*SwitchType*/ 13, MVT::i64,// ->1198
/*  1185*/   OPC_CheckPredicate4,  // Predicate_atomic_swap_64
/*  1186*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1188*/   OPC_EmitMergeInputChains1_0,
/*  1189*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XCHGD), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i64, 3/*#Ops*/, 3, 4, 2, 
             // Src: (atomic_swap:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)<<P:Predicate_atomic_swap_64>> - Complexity = 13
             // Dst: (XCHGD:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)
/*  1198*/  0, // EndSwitchType
/*  1199*/ /*SwitchOpcode*/ 43, TARGET_VAL(ISD::ATOMIC_CMP_SWAP),// ->1245
/*  1202*/  OPC_RecordMemRef,
/*  1203*/  OPC_RecordNode, // #0 = 'atomic_cmp_swap' chained node
/*  1204*/  OPC_RecordChild1, // #1 = $addr
/*  1205*/  OPC_CheckChild1TypeI64,
/*  1206*/  OPC_RecordChild2, // #2 = physreg input W0
/*  1207*/  OPC_RecordChild3, // #3 = $new
/*  1208*/  OPC_SwitchType /*2 cases */, 16, MVT::i32,// ->1227
/*  1211*/   OPC_CheckPredicate1,  // Predicate_atomic_cmp_swap_32
/*  1212*/   OPC_CheckPatternPredicate0, // (Subtarget->getHasAlu32())
/*  1213*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #4 #5
/*  1215*/   OPC_EmitMergeInputChains1_0,
/*  1216*/   OPC_EmitCopyToReg2, BPF::W0,
/*  1218*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::CMPXCHGW32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 4, 5, 3, 
             // Src: (atomic_cmp_swap:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, W0:{ *:[i32] }, GPR32:{ *:[i32] }:$new)<<P:Predicate_atomic_cmp_swap_32>> - Complexity = 13
             // Dst: (CMPXCHGW32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$new)
/*  1227*/  /*SwitchType*/ 15, MVT::i64,// ->1244
/*  1229*/   OPC_CheckPredicate4,  // Predicate_atomic_cmp_swap_64
/*  1230*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #4 #5
/*  1232*/   OPC_EmitMergeInputChains1_0,
/*  1233*/   OPC_EmitCopyToReg2, BPF::R0,
/*  1235*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::CMPXCHGD), 0|OPFL_Chain|OPFL_GlueInput|OPFL_MemRefs,
                 MVT::i64, 3/*#Ops*/, 4, 5, 3, 
             // Src: (atomic_cmp_swap:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, R0:{ *:[i64] }, GPR:{ *:[i64] }:$new)<<P:Predicate_atomic_cmp_swap_64>> - Complexity = 13
             // Dst: (CMPXCHGD:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$new)
/*  1244*/  0, // EndSwitchType
/*  1245*/ /*SwitchOpcode*/ 49, TARGET_VAL(ISD::ATOMIC_LOAD_SUB),// ->1297
/*  1248*/  OPC_RecordMemRef,
/*  1249*/  OPC_RecordNode, // #0 = 'atomic_load_sub' chained node
/*  1250*/  OPC_RecordChild1, // #1 = $addr
/*  1251*/  OPC_CheckChild1TypeI64,
/*  1252*/  OPC_RecordChild2, // #2 = $val
/*  1253*/  OPC_SwitchType /*2 cases */, 19, MVT::i32,// ->1275
/*  1256*/   OPC_CheckPredicate1,  // Predicate_atomic_load_sub_32
/*  1257*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1259*/   OPC_EmitMergeInputChains1_0,
/*  1260*/   OPC_EmitNode1None, TARGET_VAL(BPF::NEG_32),
                 MVT::i32, 1/*#Ops*/, 2,  // Results = #5
/*  1266*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFADDW32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 3, 4, 5, 
             // Src: (atomic_load_sub:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, GPR32:{ *:[i32] }:$val)<<P:Predicate_atomic_load_sub_32>> - Complexity = 13
             // Dst: (XFADDW32:{ *:[i32] } ADDRri:{ *:[i64] }:$addr, (NEG_32:{ *:[i32] } GPR32:{ *:[i32] }:$val))
/*  1275*/  /*SwitchType*/ 19, MVT::i64,// ->1296
/*  1277*/   OPC_CheckPredicate4,  // Predicate_atomic_load_sub_64
/*  1278*/   OPC_CheckComplexPat0, /*#*/1, // SelectAddr:$addr #3 #4
/*  1280*/   OPC_EmitMergeInputChains1_0,
/*  1281*/   OPC_EmitNode1None, TARGET_VAL(BPF::NEG_64),
                 MVT::i64, 1/*#Ops*/, 2,  // Results = #5
/*  1287*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::XFADDD), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i64, 3/*#Ops*/, 3, 4, 5, 
             // Src: (atomic_load_sub:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, GPR:{ *:[i64] }:$val)<<P:Predicate_atomic_load_sub_64>> - Complexity = 13
             // Dst: (XFADDD:{ *:[i64] } ADDRri:{ *:[i64] }:$addr, (NEG_64:{ *:[i64] } GPR:{ *:[i64] }:$val))
/*  1296*/  0, // EndSwitchType
/*  1297*/ /*SwitchOpcode*/ 127|128,6/*895*/, TARGET_VAL(BPFISD::BR_CC),// ->2196
/*  1301*/  OPC_RecordNode, // #0 = 'BPFbrcc' chained node
/*  1302*/  OPC_CaptureGlueInput,
/*  1303*/  OPC_RecordChild1, // #1 = $dst
/*  1304*/  OPC_Scope, 59|128,3/*443*/, /*->1750*/ // 2 children in Scope
/*  1307*/   OPC_CheckChild1TypeI64,
/*  1308*/   OPC_RecordChild2, // #2 = $imm
/*  1309*/   OPC_Scope, 74|128,1/*202*/, /*->1514*/ // 4 children in Scope
/*  1312*/    OPC_MoveChild2,
/*  1313*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1316*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  1317*/    OPC_MoveSibling3,
/*  1318*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1321*/    OPC_CheckTypeI64,
/*  1322*/    OPC_Scope, 18, /*->1342*/ // 10 children in Scope
/*  1324*/     OPC_CheckPredicate, 10, // Predicate_BPF_CC_EQ
/*  1326*/     OPC_MoveSibling4,
/*  1327*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1330*/     OPC_RecordNode, // #3 = $BrDst
/*  1331*/     OPC_MoveParent,
/*  1332*/     OPC_EmitMergeInputChains1_0,
/*  1333*/     OPC_EmitConvertToTarget2,
/*  1334*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JEQ_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_EQ>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JEQ_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1342*/    /*Scope*/ 18, /*->1361*/
/*  1343*/     OPC_CheckPredicate, 11, // Predicate_BPF_CC_GTU
/*  1345*/     OPC_MoveSibling4,
/*  1346*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1349*/     OPC_RecordNode, // #3 = $BrDst
/*  1350*/     OPC_MoveParent,
/*  1351*/     OPC_EmitMergeInputChains1_0,
/*  1352*/     OPC_EmitConvertToTarget2,
/*  1353*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JUGT_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_GTU>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JUGT_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1361*/    /*Scope*/ 18, /*->1380*/
/*  1362*/     OPC_CheckPredicate, 12, // Predicate_BPF_CC_GEU
/*  1364*/     OPC_MoveSibling4,
/*  1365*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1368*/     OPC_RecordNode, // #3 = $BrDst
/*  1369*/     OPC_MoveParent,
/*  1370*/     OPC_EmitMergeInputChains1_0,
/*  1371*/     OPC_EmitConvertToTarget2,
/*  1372*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JUGE_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_GEU>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JUGE_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1380*/    /*Scope*/ 18, /*->1399*/
/*  1381*/     OPC_CheckPredicate, 13, // Predicate_BPF_CC_NE
/*  1383*/     OPC_MoveSibling4,
/*  1384*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1387*/     OPC_RecordNode, // #3 = $BrDst
/*  1388*/     OPC_MoveParent,
/*  1389*/     OPC_EmitMergeInputChains1_0,
/*  1390*/     OPC_EmitConvertToTarget2,
/*  1391*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JNE_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_NE>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JNE_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1399*/    /*Scope*/ 18, /*->1418*/
/*  1400*/     OPC_CheckPredicate, 14, // Predicate_BPF_CC_GT
/*  1402*/     OPC_MoveSibling4,
/*  1403*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1406*/     OPC_RecordNode, // #3 = $BrDst
/*  1407*/     OPC_MoveParent,
/*  1408*/     OPC_EmitMergeInputChains1_0,
/*  1409*/     OPC_EmitConvertToTarget2,
/*  1410*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSGT_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_GT>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JSGT_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1418*/    /*Scope*/ 18, /*->1437*/
/*  1419*/     OPC_CheckPredicate, 15, // Predicate_BPF_CC_GE
/*  1421*/     OPC_MoveSibling4,
/*  1422*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1425*/     OPC_RecordNode, // #3 = $BrDst
/*  1426*/     OPC_MoveParent,
/*  1427*/     OPC_EmitMergeInputChains1_0,
/*  1428*/     OPC_EmitConvertToTarget2,
/*  1429*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSGE_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_GE>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JSGE_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1437*/    /*Scope*/ 18, /*->1456*/
/*  1438*/     OPC_CheckPredicate, 16, // Predicate_BPF_CC_LTU
/*  1440*/     OPC_MoveSibling4,
/*  1441*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1444*/     OPC_RecordNode, // #3 = $BrDst
/*  1445*/     OPC_MoveParent,
/*  1446*/     OPC_EmitMergeInputChains1_0,
/*  1447*/     OPC_EmitConvertToTarget2,
/*  1448*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JULT_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_LTU>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JULT_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1456*/    /*Scope*/ 18, /*->1475*/
/*  1457*/     OPC_CheckPredicate, 17, // Predicate_BPF_CC_LEU
/*  1459*/     OPC_MoveSibling4,
/*  1460*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1463*/     OPC_RecordNode, // #3 = $BrDst
/*  1464*/     OPC_MoveParent,
/*  1465*/     OPC_EmitMergeInputChains1_0,
/*  1466*/     OPC_EmitConvertToTarget2,
/*  1467*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JULE_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_LEU>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JULE_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1475*/    /*Scope*/ 18, /*->1494*/
/*  1476*/     OPC_CheckPredicate, 18, // Predicate_BPF_CC_LT
/*  1478*/     OPC_MoveSibling4,
/*  1479*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1482*/     OPC_RecordNode, // #3 = $BrDst
/*  1483*/     OPC_MoveParent,
/*  1484*/     OPC_EmitMergeInputChains1_0,
/*  1485*/     OPC_EmitConvertToTarget2,
/*  1486*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSLT_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_LT>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JSLT_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1494*/    /*Scope*/ 18, /*->1513*/
/*  1495*/     OPC_CheckPredicate, 19, // Predicate_BPF_CC_LE
/*  1497*/     OPC_MoveSibling4,
/*  1498*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1501*/     OPC_RecordNode, // #3 = $BrDst
/*  1502*/     OPC_MoveParent,
/*  1503*/     OPC_EmitMergeInputChains1_0,
/*  1504*/     OPC_EmitConvertToTarget2,
/*  1505*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSLE_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_LE>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JSLE_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1513*/    0, /*End of Scope*/
/*  1514*/   /*Scope*/ 59|128,1/*187*/, /*->1703*/
/*  1516*/    OPC_MoveChild3,
/*  1517*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1520*/    OPC_CheckTypeI64,
/*  1521*/    OPC_Scope, 17, /*->1540*/ // 10 children in Scope
/*  1523*/     OPC_CheckPredicate, 10, // Predicate_BPF_CC_EQ
/*  1525*/     OPC_MoveSibling4,
/*  1526*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1529*/     OPC_RecordNode, // #3 = $BrDst
/*  1530*/     OPC_MoveParent,
/*  1531*/     OPC_EmitMergeInputChains1_0,
/*  1532*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JEQ_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_EQ>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JEQ_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1540*/    /*Scope*/ 17, /*->1558*/
/*  1541*/     OPC_CheckPredicate, 11, // Predicate_BPF_CC_GTU
/*  1543*/     OPC_MoveSibling4,
/*  1544*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1547*/     OPC_RecordNode, // #3 = $BrDst
/*  1548*/     OPC_MoveParent,
/*  1549*/     OPC_EmitMergeInputChains1_0,
/*  1550*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JUGT_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_GTU>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JUGT_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1558*/    /*Scope*/ 17, /*->1576*/
/*  1559*/     OPC_CheckPredicate, 12, // Predicate_BPF_CC_GEU
/*  1561*/     OPC_MoveSibling4,
/*  1562*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1565*/     OPC_RecordNode, // #3 = $BrDst
/*  1566*/     OPC_MoveParent,
/*  1567*/     OPC_EmitMergeInputChains1_0,
/*  1568*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JUGE_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_GEU>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JUGE_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1576*/    /*Scope*/ 17, /*->1594*/
/*  1577*/     OPC_CheckPredicate, 13, // Predicate_BPF_CC_NE
/*  1579*/     OPC_MoveSibling4,
/*  1580*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1583*/     OPC_RecordNode, // #3 = $BrDst
/*  1584*/     OPC_MoveParent,
/*  1585*/     OPC_EmitMergeInputChains1_0,
/*  1586*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JNE_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_NE>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JNE_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1594*/    /*Scope*/ 17, /*->1612*/
/*  1595*/     OPC_CheckPredicate, 14, // Predicate_BPF_CC_GT
/*  1597*/     OPC_MoveSibling4,
/*  1598*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1601*/     OPC_RecordNode, // #3 = $BrDst
/*  1602*/     OPC_MoveParent,
/*  1603*/     OPC_EmitMergeInputChains1_0,
/*  1604*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSGT_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_GT>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JSGT_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1612*/    /*Scope*/ 17, /*->1630*/
/*  1613*/     OPC_CheckPredicate, 15, // Predicate_BPF_CC_GE
/*  1615*/     OPC_MoveSibling4,
/*  1616*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1619*/     OPC_RecordNode, // #3 = $BrDst
/*  1620*/     OPC_MoveParent,
/*  1621*/     OPC_EmitMergeInputChains1_0,
/*  1622*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSGE_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_GE>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JSGE_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1630*/    /*Scope*/ 17, /*->1648*/
/*  1631*/     OPC_CheckPredicate, 16, // Predicate_BPF_CC_LTU
/*  1633*/     OPC_MoveSibling4,
/*  1634*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1637*/     OPC_RecordNode, // #3 = $BrDst
/*  1638*/     OPC_MoveParent,
/*  1639*/     OPC_EmitMergeInputChains1_0,
/*  1640*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JULT_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_LTU>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JULT_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1648*/    /*Scope*/ 17, /*->1666*/
/*  1649*/     OPC_CheckPredicate, 17, // Predicate_BPF_CC_LEU
/*  1651*/     OPC_MoveSibling4,
/*  1652*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1655*/     OPC_RecordNode, // #3 = $BrDst
/*  1656*/     OPC_MoveParent,
/*  1657*/     OPC_EmitMergeInputChains1_0,
/*  1658*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JULE_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_LEU>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JULE_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1666*/    /*Scope*/ 17, /*->1684*/
/*  1667*/     OPC_CheckPredicate, 18, // Predicate_BPF_CC_LT
/*  1669*/     OPC_MoveSibling4,
/*  1670*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1673*/     OPC_RecordNode, // #3 = $BrDst
/*  1674*/     OPC_MoveParent,
/*  1675*/     OPC_EmitMergeInputChains1_0,
/*  1676*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSLT_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_LT>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JSLT_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1684*/    /*Scope*/ 17, /*->1702*/
/*  1685*/     OPC_CheckPredicate, 19, // Predicate_BPF_CC_LE
/*  1687*/     OPC_MoveSibling4,
/*  1688*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1691*/     OPC_RecordNode, // #3 = $BrDst
/*  1692*/     OPC_MoveParent,
/*  1693*/     OPC_EmitMergeInputChains1_0,
/*  1694*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSLE_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (imm:{ *:[i64] })<<P:Predicate_BPF_CC_LE>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JSLE_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1702*/    0, /*End of Scope*/
/*  1703*/   /*Scope*/ 25, /*->1729*/
/*  1704*/    OPC_MoveChild2,
/*  1705*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1708*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  1709*/    OPC_MoveSibling3,
/*  1710*/    OPC_CheckOpcode, TARGET_VAL(ISD::VALUETYPE),
/*  1713*/    OPC_MoveSibling4,
/*  1714*/    OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1717*/    OPC_RecordNode, // #3 = $BrDst
/*  1718*/    OPC_MoveParent,
/*  1719*/    OPC_EmitMergeInputChains1_0,
/*  1720*/    OPC_EmitConvertToTarget2,
/*  1721*/    OPC_MorphNodeTo0, TARGET_VAL(BPF::JSET_ri), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                  3/*#Ops*/, 1, 4, 3, 
              // Src: (BPFbrcc i64:{ *:[i64] }:$dst, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm, (vt:{ *:[Other] }), (bb:{ *:[Other] }):$BrDst) - Complexity = 7
              // Dst: (JSET_ri i64:{ *:[i64] }:$dst, (imm:{ *:[i64] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1729*/   /*Scope*/ 19, /*->1749*/
/*  1730*/    OPC_MoveChild3,
/*  1731*/    OPC_CheckOpcode, TARGET_VAL(ISD::VALUETYPE),
/*  1734*/    OPC_MoveSibling4,
/*  1735*/    OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1738*/    OPC_RecordNode, // #3 = $BrDst
/*  1739*/    OPC_MoveParent,
/*  1740*/    OPC_EmitMergeInputChains1_0,
/*  1741*/    OPC_MorphNodeTo0, TARGET_VAL(BPF::JSET_rr), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                  3/*#Ops*/, 1, 2, 3, 
              // Src: (BPFbrcc i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (vt:{ *:[Other] }), (bb:{ *:[Other] }):$BrDst) - Complexity = 3
              // Dst: (JSET_rr i64:{ *:[i64] }:$dst, i64:{ *:[i64] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1749*/   0, /*End of Scope*/
/*  1750*/  /*Scope*/ 59|128,3/*443*/, /*->2195*/
/*  1752*/   OPC_CheckChild1TypeI32,
/*  1753*/   OPC_RecordChild2, // #2 = $imm
/*  1754*/   OPC_Scope, 74|128,1/*202*/, /*->1959*/ // 4 children in Scope
/*  1757*/    OPC_MoveChild2,
/*  1758*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1761*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  1762*/    OPC_MoveSibling3,
/*  1763*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1766*/    OPC_CheckTypeI32,
/*  1767*/    OPC_Scope, 18, /*->1787*/ // 10 children in Scope
/*  1769*/     OPC_CheckPredicate, 10, // Predicate_BPF_CC_EQ_32
/*  1771*/     OPC_MoveSibling4,
/*  1772*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1775*/     OPC_RecordNode, // #3 = $BrDst
/*  1776*/     OPC_MoveParent,
/*  1777*/     OPC_EmitMergeInputChains1_0,
/*  1778*/     OPC_EmitConvertToTarget2,
/*  1779*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JEQ_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_EQ_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JEQ_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1787*/    /*Scope*/ 18, /*->1806*/
/*  1788*/     OPC_CheckPredicate, 11, // Predicate_BPF_CC_GTU_32
/*  1790*/     OPC_MoveSibling4,
/*  1791*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1794*/     OPC_RecordNode, // #3 = $BrDst
/*  1795*/     OPC_MoveParent,
/*  1796*/     OPC_EmitMergeInputChains1_0,
/*  1797*/     OPC_EmitConvertToTarget2,
/*  1798*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JUGT_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_GTU_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JUGT_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1806*/    /*Scope*/ 18, /*->1825*/
/*  1807*/     OPC_CheckPredicate, 12, // Predicate_BPF_CC_GEU_32
/*  1809*/     OPC_MoveSibling4,
/*  1810*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1813*/     OPC_RecordNode, // #3 = $BrDst
/*  1814*/     OPC_MoveParent,
/*  1815*/     OPC_EmitMergeInputChains1_0,
/*  1816*/     OPC_EmitConvertToTarget2,
/*  1817*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JUGE_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_GEU_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JUGE_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1825*/    /*Scope*/ 18, /*->1844*/
/*  1826*/     OPC_CheckPredicate, 13, // Predicate_BPF_CC_NE_32
/*  1828*/     OPC_MoveSibling4,
/*  1829*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1832*/     OPC_RecordNode, // #3 = $BrDst
/*  1833*/     OPC_MoveParent,
/*  1834*/     OPC_EmitMergeInputChains1_0,
/*  1835*/     OPC_EmitConvertToTarget2,
/*  1836*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JNE_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_NE_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JNE_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1844*/    /*Scope*/ 18, /*->1863*/
/*  1845*/     OPC_CheckPredicate, 14, // Predicate_BPF_CC_GT_32
/*  1847*/     OPC_MoveSibling4,
/*  1848*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1851*/     OPC_RecordNode, // #3 = $BrDst
/*  1852*/     OPC_MoveParent,
/*  1853*/     OPC_EmitMergeInputChains1_0,
/*  1854*/     OPC_EmitConvertToTarget2,
/*  1855*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSGT_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_GT_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JSGT_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1863*/    /*Scope*/ 18, /*->1882*/
/*  1864*/     OPC_CheckPredicate, 15, // Predicate_BPF_CC_GE_32
/*  1866*/     OPC_MoveSibling4,
/*  1867*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1870*/     OPC_RecordNode, // #3 = $BrDst
/*  1871*/     OPC_MoveParent,
/*  1872*/     OPC_EmitMergeInputChains1_0,
/*  1873*/     OPC_EmitConvertToTarget2,
/*  1874*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSGE_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_GE_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JSGE_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1882*/    /*Scope*/ 18, /*->1901*/
/*  1883*/     OPC_CheckPredicate, 16, // Predicate_BPF_CC_LTU_32
/*  1885*/     OPC_MoveSibling4,
/*  1886*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1889*/     OPC_RecordNode, // #3 = $BrDst
/*  1890*/     OPC_MoveParent,
/*  1891*/     OPC_EmitMergeInputChains1_0,
/*  1892*/     OPC_EmitConvertToTarget2,
/*  1893*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JULT_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_LTU_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JULT_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1901*/    /*Scope*/ 18, /*->1920*/
/*  1902*/     OPC_CheckPredicate, 17, // Predicate_BPF_CC_LEU_32
/*  1904*/     OPC_MoveSibling4,
/*  1905*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1908*/     OPC_RecordNode, // #3 = $BrDst
/*  1909*/     OPC_MoveParent,
/*  1910*/     OPC_EmitMergeInputChains1_0,
/*  1911*/     OPC_EmitConvertToTarget2,
/*  1912*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JULE_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_LEU_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JULE_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1920*/    /*Scope*/ 18, /*->1939*/
/*  1921*/     OPC_CheckPredicate, 18, // Predicate_BPF_CC_LT_32
/*  1923*/     OPC_MoveSibling4,
/*  1924*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1927*/     OPC_RecordNode, // #3 = $BrDst
/*  1928*/     OPC_MoveParent,
/*  1929*/     OPC_EmitMergeInputChains1_0,
/*  1930*/     OPC_EmitConvertToTarget2,
/*  1931*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSLT_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_LT_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JSLT_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1939*/    /*Scope*/ 18, /*->1958*/
/*  1940*/     OPC_CheckPredicate, 19, // Predicate_BPF_CC_LE_32
/*  1942*/     OPC_MoveSibling4,
/*  1943*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1946*/     OPC_RecordNode, // #3 = $BrDst
/*  1947*/     OPC_MoveParent,
/*  1948*/     OPC_EmitMergeInputChains1_0,
/*  1949*/     OPC_EmitConvertToTarget2,
/*  1950*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSLE_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 4, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_LE_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 11
               // Dst: (JSLE_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  1958*/    0, /*End of Scope*/
/*  1959*/   /*Scope*/ 59|128,1/*187*/, /*->2148*/
/*  1961*/    OPC_MoveChild3,
/*  1962*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1965*/    OPC_CheckTypeI32,
/*  1966*/    OPC_Scope, 17, /*->1985*/ // 10 children in Scope
/*  1968*/     OPC_CheckPredicate, 10, // Predicate_BPF_CC_EQ_32
/*  1970*/     OPC_MoveSibling4,
/*  1971*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1974*/     OPC_RecordNode, // #3 = $BrDst
/*  1975*/     OPC_MoveParent,
/*  1976*/     OPC_EmitMergeInputChains1_0,
/*  1977*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JEQ_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_EQ_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JEQ_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  1985*/    /*Scope*/ 17, /*->2003*/
/*  1986*/     OPC_CheckPredicate, 11, // Predicate_BPF_CC_GTU_32
/*  1988*/     OPC_MoveSibling4,
/*  1989*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1992*/     OPC_RecordNode, // #3 = $BrDst
/*  1993*/     OPC_MoveParent,
/*  1994*/     OPC_EmitMergeInputChains1_0,
/*  1995*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JUGT_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_GTU_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JUGT_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2003*/    /*Scope*/ 17, /*->2021*/
/*  2004*/     OPC_CheckPredicate, 12, // Predicate_BPF_CC_GEU_32
/*  2006*/     OPC_MoveSibling4,
/*  2007*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2010*/     OPC_RecordNode, // #3 = $BrDst
/*  2011*/     OPC_MoveParent,
/*  2012*/     OPC_EmitMergeInputChains1_0,
/*  2013*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JUGE_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_GEU_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JUGE_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2021*/    /*Scope*/ 17, /*->2039*/
/*  2022*/     OPC_CheckPredicate, 13, // Predicate_BPF_CC_NE_32
/*  2024*/     OPC_MoveSibling4,
/*  2025*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2028*/     OPC_RecordNode, // #3 = $BrDst
/*  2029*/     OPC_MoveParent,
/*  2030*/     OPC_EmitMergeInputChains1_0,
/*  2031*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JNE_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_NE_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JNE_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2039*/    /*Scope*/ 17, /*->2057*/
/*  2040*/     OPC_CheckPredicate, 14, // Predicate_BPF_CC_GT_32
/*  2042*/     OPC_MoveSibling4,
/*  2043*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2046*/     OPC_RecordNode, // #3 = $BrDst
/*  2047*/     OPC_MoveParent,
/*  2048*/     OPC_EmitMergeInputChains1_0,
/*  2049*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSGT_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_GT_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JSGT_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2057*/    /*Scope*/ 17, /*->2075*/
/*  2058*/     OPC_CheckPredicate, 15, // Predicate_BPF_CC_GE_32
/*  2060*/     OPC_MoveSibling4,
/*  2061*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2064*/     OPC_RecordNode, // #3 = $BrDst
/*  2065*/     OPC_MoveParent,
/*  2066*/     OPC_EmitMergeInputChains1_0,
/*  2067*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSGE_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_GE_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JSGE_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2075*/    /*Scope*/ 17, /*->2093*/
/*  2076*/     OPC_CheckPredicate, 16, // Predicate_BPF_CC_LTU_32
/*  2078*/     OPC_MoveSibling4,
/*  2079*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2082*/     OPC_RecordNode, // #3 = $BrDst
/*  2083*/     OPC_MoveParent,
/*  2084*/     OPC_EmitMergeInputChains1_0,
/*  2085*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JULT_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_LTU_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JULT_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2093*/    /*Scope*/ 17, /*->2111*/
/*  2094*/     OPC_CheckPredicate, 17, // Predicate_BPF_CC_LEU_32
/*  2096*/     OPC_MoveSibling4,
/*  2097*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2100*/     OPC_RecordNode, // #3 = $BrDst
/*  2101*/     OPC_MoveParent,
/*  2102*/     OPC_EmitMergeInputChains1_0,
/*  2103*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JULE_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_LEU_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JULE_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2111*/    /*Scope*/ 17, /*->2129*/
/*  2112*/     OPC_CheckPredicate, 18, // Predicate_BPF_CC_LT_32
/*  2114*/     OPC_MoveSibling4,
/*  2115*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2118*/     OPC_RecordNode, // #3 = $BrDst
/*  2119*/     OPC_MoveParent,
/*  2120*/     OPC_EmitMergeInputChains1_0,
/*  2121*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSLT_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_LT_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JSLT_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2129*/    /*Scope*/ 17, /*->2147*/
/*  2130*/     OPC_CheckPredicate, 19, // Predicate_BPF_CC_LE_32
/*  2132*/     OPC_MoveSibling4,
/*  2133*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2136*/     OPC_RecordNode, // #3 = $BrDst
/*  2137*/     OPC_MoveParent,
/*  2138*/     OPC_EmitMergeInputChains1_0,
/*  2139*/     OPC_MorphNodeTo0, TARGET_VAL(BPF::JSLE_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                   3/*#Ops*/, 1, 2, 3, 
               // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_BPF_CC_LE_32>>, (bb:{ *:[Other] }):$BrDst) - Complexity = 7
               // Dst: (JSLE_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2147*/    0, /*End of Scope*/
/*  2148*/   /*Scope*/ 25, /*->2174*/
/*  2149*/    OPC_MoveChild2,
/*  2150*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2153*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2154*/    OPC_MoveSibling3,
/*  2155*/    OPC_CheckOpcode, TARGET_VAL(ISD::VALUETYPE),
/*  2158*/    OPC_MoveSibling4,
/*  2159*/    OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2162*/    OPC_RecordNode, // #3 = $BrDst
/*  2163*/    OPC_MoveParent,
/*  2164*/    OPC_EmitMergeInputChains1_0,
/*  2165*/    OPC_EmitConvertToTarget2,
/*  2166*/    OPC_MorphNodeTo0, TARGET_VAL(BPF::JSET_ri_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                  3/*#Ops*/, 1, 4, 3, 
              // Src: (BPFbrcc i32:{ *:[i32] }:$dst, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm, (vt:{ *:[Other] }), (bb:{ *:[Other] }):$BrDst) - Complexity = 7
              // Dst: (JSET_ri_32 i32:{ *:[i32] }:$dst, (imm:{ *:[i32] }):$imm, (bb:{ *:[Other] }):$BrDst)
/*  2174*/   /*Scope*/ 19, /*->2194*/
/*  2175*/    OPC_MoveChild3,
/*  2176*/    OPC_CheckOpcode, TARGET_VAL(ISD::VALUETYPE),
/*  2179*/    OPC_MoveSibling4,
/*  2180*/    OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  2183*/    OPC_RecordNode, // #3 = $BrDst
/*  2184*/    OPC_MoveParent,
/*  2185*/    OPC_EmitMergeInputChains1_0,
/*  2186*/    OPC_MorphNodeTo0, TARGET_VAL(BPF::JSET_rr_32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                  3/*#Ops*/, 1, 2, 3, 
              // Src: (BPFbrcc i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (vt:{ *:[Other] }), (bb:{ *:[Other] }):$BrDst) - Complexity = 3
              // Dst: (JSET_rr_32 i32:{ *:[i32] }:$dst, i32:{ *:[i32] }:$src, (bb:{ *:[Other] }):$BrDst)
/*  2194*/   0, /*End of Scope*/
/*  2195*/  0, /*End of Scope*/
/*  2196*/ /*SwitchOpcode*/ 6|128,1/*134*/, TARGET_VAL(ISD::SRL),// ->2334
/*  2200*/  OPC_Scope, 73, /*->2275*/ // 2 children in Scope
/*  2202*/   OPC_MoveChild0,
/*  2203*/   OPC_CheckOpcode, TARGET_VAL(ISD::BSWAP),
/*  2206*/   OPC_RecordChild0, // #0 = $src
/*  2207*/   OPC_MoveParent,
/*  2208*/   OPC_CheckTypeI64,
/*  2209*/   OPC_Scope, 31, /*->2242*/ // 2 children in Scope
/*  2211*/    OPC_CheckChild1Integer, 96, 
/*  2213*/    OPC_CheckChild1TypeI64,
/*  2214*/    OPC_Scope, 7, /*->2223*/ // 3 children in Scope
/*  2216*/     OPC_CheckPatternPredicate7, // (Subtarget->hasBswap())
/*  2217*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::BSWAP16),
                   MVT::i64, 1/*#Ops*/, 0, 
               // Src: (srl:{ *:[i64] } (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src), 48:{ *:[i64] }) - Complexity = 11
               // Dst: (BSWAP16:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  2223*/    /*Scope*/ 8, /*->2232*/
/*  2224*/     OPC_CheckPatternPredicate, 8, // (Subtarget->isLittleEndian())
/*  2226*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::BE16),
                   MVT::i64, 1/*#Ops*/, 0, 
               // Src: (srl:{ *:[i64] } (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src), 48:{ *:[i64] }) - Complexity = 11
               // Dst: (BE16:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  2232*/    /*Scope*/ 8, /*->2241*/
/*  2233*/     OPC_CheckPatternPredicate, 9, // (!Subtarget->isLittleEndian())
/*  2235*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::LE16),
                   MVT::i64, 1/*#Ops*/, 0, 
               // Src: (srl:{ *:[i64] } (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src), 48:{ *:[i64] }) - Complexity = 11
               // Dst: (LE16:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  2241*/    0, /*End of Scope*/
/*  2242*/   /*Scope*/ 31, /*->2274*/
/*  2243*/    OPC_CheckChild1Integer, 64, 
/*  2245*/    OPC_CheckChild1TypeI64,
/*  2246*/    OPC_Scope, 7, /*->2255*/ // 3 children in Scope
/*  2248*/     OPC_CheckPatternPredicate7, // (Subtarget->hasBswap())
/*  2249*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::BSWAP32),
                   MVT::i64, 1/*#Ops*/, 0, 
               // Src: (srl:{ *:[i64] } (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src), 32:{ *:[i64] }) - Complexity = 11
               // Dst: (BSWAP32:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  2255*/    /*Scope*/ 8, /*->2264*/
/*  2256*/     OPC_CheckPatternPredicate, 8, // (Subtarget->isLittleEndian())
/*  2258*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::BE32),
                   MVT::i64, 1/*#Ops*/, 0, 
               // Src: (srl:{ *:[i64] } (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src), 32:{ *:[i64] }) - Complexity = 11
               // Dst: (BE32:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  2264*/    /*Scope*/ 8, /*->2273*/
/*  2265*/     OPC_CheckPatternPredicate, 9, // (!Subtarget->isLittleEndian())
/*  2267*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::LE32),
                   MVT::i64, 1/*#Ops*/, 0, 
               // Src: (srl:{ *:[i64] } (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src), 32:{ *:[i64] }) - Complexity = 11
               // Dst: (LE32:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  2273*/    0, /*End of Scope*/
/*  2274*/   0, /*End of Scope*/
/*  2275*/  /*Scope*/ 57, /*->2333*/
/*  2276*/   OPC_RecordChild0, // #0 = $src2
/*  2277*/   OPC_RecordChild1, // #1 = $imm
/*  2278*/   OPC_Scope, 32, /*->2312*/ // 3 children in Scope
/*  2280*/    OPC_MoveChild1,
/*  2281*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2284*/    OPC_SwitchType /*2 cases */, 11, MVT::i64,// ->2298
/*  2287*/     OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  2288*/     OPC_MoveParent,
/*  2289*/     OPC_CheckTypeI64,
/*  2290*/     OPC_EmitConvertToTarget1,
/*  2291*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRL_ri),
                   MVT::i64, 2/*#Ops*/, 0, 2, 
               // Src: (srl:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
               // Dst: (SRL_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  2298*/    /*SwitchType*/ 11, MVT::i32,// ->2311
/*  2300*/     OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2301*/     OPC_MoveParent,
/*  2302*/     OPC_CheckTypeI32,
/*  2303*/     OPC_EmitConvertToTarget1,
/*  2304*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRL_ri_32),
                   MVT::i32, 2/*#Ops*/, 0, 2, 
               // Src: (srl:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
               // Dst: (SRL_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  2311*/    0, // EndSwitchType
/*  2312*/   /*Scope*/ 9, /*->2322*/
/*  2313*/    OPC_CheckChild1TypeI64,
/*  2314*/    OPC_CheckTypeI64,
/*  2315*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRL_rr),
                  MVT::i64, 2/*#Ops*/, 0, 1, 
              // Src: (srl:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
              // Dst: (SRL_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  2322*/   /*Scope*/ 9, /*->2332*/
/*  2323*/    OPC_CheckChild1TypeI32,
/*  2324*/    OPC_CheckTypeI32,
/*  2325*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRL_rr_32),
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (srl:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
              // Dst: (SRL_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  2332*/   0, /*End of Scope*/
/*  2333*/  0, /*End of Scope*/
/*  2334*/ /*SwitchOpcode*/ 52|128,1/*180*/, TARGET_VAL(BPFISD::SELECT_CC),// ->2518
/*  2338*/  OPC_CaptureGlueInput,
/*  2339*/  OPC_RecordChild0, // #0 = $lhs
/*  2340*/  OPC_Scope, 87, /*->2429*/ // 2 children in Scope
/*  2342*/   OPC_CheckChild0TypeI64,
/*  2343*/   OPC_RecordChild1, // #1 = $rhs
/*  2344*/   OPC_Scope, 44, /*->2390*/ // 2 children in Scope
/*  2346*/    OPC_MoveChild1,
/*  2347*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2350*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  2351*/    OPC_MoveSibling2,
/*  2352*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2355*/    OPC_RecordNode, // #2 = $imm
/*  2356*/    OPC_CheckTypeI64,
/*  2357*/    OPC_MoveParent,
/*  2358*/    OPC_RecordChild3, // #3 = $src
/*  2359*/    OPC_RecordChild4, // #4 = $src2
/*  2360*/    OPC_SwitchType /*2 cases */, 12, MVT::i64,// ->2375
/*  2363*/     OPC_EmitConvertToTarget1,
/*  2364*/     OPC_EmitConvertToTarget2,
/*  2365*/     OPC_MorphNodeTo1GlueInput, TARGET_VAL(BPF::Select_Ri),
                   MVT::i64, 5/*#Ops*/, 0, 5, 6, 3, 4, 
               // Src: (BPFselectcc:{ *:[i64] } i64:{ *:[i64] }:$lhs, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$rhs, (imm:{ *:[i64] }):$imm, i64:{ *:[i64] }:$src, i64:{ *:[i64] }:$src2) - Complexity = 10
               // Dst: (Select_Ri:{ *:[i64] } i64:{ *:[i64] }:$lhs, (imm:{ *:[i64] }):$rhs, (imm:{ *:[i64] }):$imm, i64:{ *:[i64] }:$src, i64:{ *:[i64] }:$src2)
/*  2375*/    /*SwitchType*/ 12, MVT::i32,// ->2389
/*  2377*/     OPC_EmitConvertToTarget1,
/*  2378*/     OPC_EmitConvertToTarget2,
/*  2379*/     OPC_MorphNodeTo1GlueInput, TARGET_VAL(BPF::Select_Ri_64_32),
                   MVT::i32, 5/*#Ops*/, 0, 5, 6, 3, 4, 
               // Src: (BPFselectcc:{ *:[i32] } i64:{ *:[i64] }:$lhs, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$rhs, (imm:{ *:[i64] }):$imm, i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src2) - Complexity = 10
               // Dst: (Select_Ri_64_32:{ *:[i32] } i64:{ *:[i64] }:$lhs, (imm:{ *:[i64] }):$rhs, (imm:{ *:[i64] }):$imm, i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src2)
/*  2389*/    0, // EndSwitchType
/*  2390*/   /*Scope*/ 37, /*->2428*/
/*  2391*/    OPC_RecordChild2, // #2 = $imm
/*  2392*/    OPC_MoveChild2,
/*  2393*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2396*/    OPC_CheckTypeI64,
/*  2397*/    OPC_MoveParent,
/*  2398*/    OPC_RecordChild3, // #3 = $src
/*  2399*/    OPC_RecordChild4, // #4 = $src2
/*  2400*/    OPC_SwitchType /*2 cases */, 11, MVT::i64,// ->2414
/*  2403*/     OPC_EmitConvertToTarget2,
/*  2404*/     OPC_MorphNodeTo1GlueInput, TARGET_VAL(BPF::Select),
                   MVT::i64, 5/*#Ops*/, 0, 1, 5, 3, 4, 
               // Src: (BPFselectcc:{ *:[i64] } i64:{ *:[i64] }:$lhs, i64:{ *:[i64] }:$rhs, (imm:{ *:[i64] }):$imm, i64:{ *:[i64] }:$src, i64:{ *:[i64] }:$src2) - Complexity = 6
               // Dst: (Select:{ *:[i64] } i64:{ *:[i64] }:$lhs, i64:{ *:[i64] }:$rhs, (imm:{ *:[i64] }):$imm, i64:{ *:[i64] }:$src, i64:{ *:[i64] }:$src2)
/*  2414*/    /*SwitchType*/ 11, MVT::i32,// ->2427
/*  2416*/     OPC_EmitConvertToTarget2,
/*  2417*/     OPC_MorphNodeTo1GlueInput, TARGET_VAL(BPF::Select_64_32),
                   MVT::i32, 5/*#Ops*/, 0, 1, 5, 3, 4, 
               // Src: (BPFselectcc:{ *:[i32] } i64:{ *:[i64] }:$lhs, i64:{ *:[i64] }:$rhs, (imm:{ *:[i64] }):$imm, i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src2) - Complexity = 6
               // Dst: (Select_64_32:{ *:[i32] } i64:{ *:[i64] }:$lhs, i64:{ *:[i64] }:$rhs, (imm:{ *:[i64] }):$imm, i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src2)
/*  2427*/    0, // EndSwitchType
/*  2428*/   0, /*End of Scope*/
/*  2429*/  /*Scope*/ 87, /*->2517*/
/*  2430*/   OPC_CheckChild0TypeI32,
/*  2431*/   OPC_RecordChild1, // #1 = $rhs
/*  2432*/   OPC_Scope, 44, /*->2478*/ // 2 children in Scope
/*  2434*/    OPC_MoveChild1,
/*  2435*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2438*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2439*/    OPC_MoveSibling2,
/*  2440*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2443*/    OPC_RecordNode, // #2 = $imm
/*  2444*/    OPC_CheckTypeI32,
/*  2445*/    OPC_MoveParent,
/*  2446*/    OPC_RecordChild3, // #3 = $src
/*  2447*/    OPC_RecordChild4, // #4 = $src2
/*  2448*/    OPC_SwitchType /*2 cases */, 12, MVT::i32,// ->2463
/*  2451*/     OPC_EmitConvertToTarget1,
/*  2452*/     OPC_EmitConvertToTarget2,
/*  2453*/     OPC_MorphNodeTo1GlueInput, TARGET_VAL(BPF::Select_Ri_32),
                   MVT::i32, 5/*#Ops*/, 0, 5, 6, 3, 4, 
               // Src: (BPFselectcc:{ *:[i32] } i32:{ *:[i32] }:$lhs, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$rhs, (imm:{ *:[i32] }):$imm, i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src2) - Complexity = 10
               // Dst: (Select_Ri_32:{ *:[i32] } i32:{ *:[i32] }:$lhs, (imm:{ *:[i32] }):$rhs, (imm:{ *:[i32] }):$imm, i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src2)
/*  2463*/    /*SwitchType*/ 12, MVT::i64,// ->2477
/*  2465*/     OPC_EmitConvertToTarget1,
/*  2466*/     OPC_EmitConvertToTarget2,
/*  2467*/     OPC_MorphNodeTo1GlueInput, TARGET_VAL(BPF::Select_Ri_32_64),
                   MVT::i64, 5/*#Ops*/, 0, 5, 6, 3, 4, 
               // Src: (BPFselectcc:{ *:[i64] } i32:{ *:[i32] }:$lhs, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$rhs, (imm:{ *:[i32] }):$imm, i64:{ *:[i64] }:$src, i64:{ *:[i64] }:$src2) - Complexity = 10
               // Dst: (Select_Ri_32_64:{ *:[i64] } i32:{ *:[i32] }:$lhs, (imm:{ *:[i32] }):$rhs, (imm:{ *:[i32] }):$imm, i64:{ *:[i64] }:$src, i64:{ *:[i64] }:$src2)
/*  2477*/    0, // EndSwitchType
/*  2478*/   /*Scope*/ 37, /*->2516*/
/*  2479*/    OPC_RecordChild2, // #2 = $imm
/*  2480*/    OPC_MoveChild2,
/*  2481*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2484*/    OPC_CheckTypeI32,
/*  2485*/    OPC_MoveParent,
/*  2486*/    OPC_RecordChild3, // #3 = $src
/*  2487*/    OPC_RecordChild4, // #4 = $src2
/*  2488*/    OPC_SwitchType /*2 cases */, 11, MVT::i32,// ->2502
/*  2491*/     OPC_EmitConvertToTarget2,
/*  2492*/     OPC_MorphNodeTo1GlueInput, TARGET_VAL(BPF::Select_32),
                   MVT::i32, 5/*#Ops*/, 0, 1, 5, 3, 4, 
               // Src: (BPFselectcc:{ *:[i32] } i32:{ *:[i32] }:$lhs, i32:{ *:[i32] }:$rhs, (imm:{ *:[i32] }):$imm, i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src2) - Complexity = 6
               // Dst: (Select_32:{ *:[i32] } i32:{ *:[i32] }:$lhs, i32:{ *:[i32] }:$rhs, (imm:{ *:[i32] }):$imm, i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src2)
/*  2502*/    /*SwitchType*/ 11, MVT::i64,// ->2515
/*  2504*/     OPC_EmitConvertToTarget2,
/*  2505*/     OPC_MorphNodeTo1GlueInput, TARGET_VAL(BPF::Select_32_64),
                   MVT::i64, 5/*#Ops*/, 0, 1, 5, 3, 4, 
               // Src: (BPFselectcc:{ *:[i64] } i32:{ *:[i32] }:$lhs, i32:{ *:[i32] }:$rhs, (imm:{ *:[i32] }):$imm, i64:{ *:[i64] }:$src, i64:{ *:[i64] }:$src2) - Complexity = 6
               // Dst: (Select_32_64:{ *:[i64] } i32:{ *:[i32] }:$lhs, i32:{ *:[i32] }:$rhs, (imm:{ *:[i32] }):$imm, i64:{ *:[i64] }:$src, i64:{ *:[i64] }:$src2)
/*  2515*/    0, // EndSwitchType
/*  2516*/   0, /*End of Scope*/
/*  2517*/  0, /*End of Scope*/
/*  2518*/ /*SwitchOpcode*/ 68, TARGET_VAL(ISD::ADD),// ->2589
/*  2521*/  OPC_Scope, 11, /*->2534*/ // 2 children in Scope
/*  2523*/   OPC_RecordNode, // #0 = $addr
/*  2524*/   OPC_CheckTypeI64,
/*  2525*/   OPC_CheckComplexPat1, /*#*/0, // SelectFIAddr:$addr #1 #2
/*  2527*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::FI_ri),
                 MVT::i64, 2/*#Ops*/, 1, 2, 
             // Src: FIri:{ *:[i64] }:$addr - Complexity = 9
             // Dst: (FI_ri:{ *:[i64] } FIri:{ *:[i64] }:$addr)
/*  2534*/  /*Scope*/ 53, /*->2588*/
/*  2535*/   OPC_RecordChild0, // #0 = $src2
/*  2536*/   OPC_RecordChild1, // #1 = $imm
/*  2537*/   OPC_Scope, 30, /*->2569*/ // 3 children in Scope
/*  2539*/    OPC_MoveChild1,
/*  2540*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2543*/    OPC_Scope, 11, /*->2556*/ // 2 children in Scope
/*  2545*/     OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  2546*/     OPC_MoveParent,
/*  2547*/     OPC_CheckTypeI64,
/*  2548*/     OPC_EmitConvertToTarget1,
/*  2549*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::ADD_ri),
                   MVT::i64, 2/*#Ops*/, 0, 2, 
               // Src: (add:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
               // Dst: (ADD_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  2556*/    /*Scope*/ 11, /*->2568*/
/*  2557*/     OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2558*/     OPC_MoveParent,
/*  2559*/     OPC_CheckTypeI32,
/*  2560*/     OPC_EmitConvertToTarget1,
/*  2561*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::ADD_ri_32),
                   MVT::i32, 2/*#Ops*/, 0, 2, 
               // Src: (add:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
               // Dst: (ADD_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  2568*/    0, /*End of Scope*/
/*  2569*/   /*Scope*/ 8, /*->2578*/
/*  2570*/    OPC_CheckTypeI64,
/*  2571*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::ADD_rr),
                  MVT::i64, 2/*#Ops*/, 0, 1, 
              // Src: (add:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
              // Dst: (ADD_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  2578*/   /*Scope*/ 8, /*->2587*/
/*  2579*/    OPC_CheckTypeI32,
/*  2580*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::ADD_rr_32),
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (add:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
              // Dst: (ADD_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  2587*/   0, /*End of Scope*/
/*  2588*/  0, /*End of Scope*/
/*  2589*/ /*SwitchOpcode*/ 68, TARGET_VAL(ISD::OR),// ->2660
/*  2592*/  OPC_Scope, 11, /*->2605*/ // 2 children in Scope
/*  2594*/   OPC_RecordNode, // #0 = $addr
/*  2595*/   OPC_CheckTypeI64,
/*  2596*/   OPC_CheckComplexPat1, /*#*/0, // SelectFIAddr:$addr #1 #2
/*  2598*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::FI_ri),
                 MVT::i64, 2/*#Ops*/, 1, 2, 
             // Src: FIri:{ *:[i64] }:$addr - Complexity = 9
             // Dst: (FI_ri:{ *:[i64] } FIri:{ *:[i64] }:$addr)
/*  2605*/  /*Scope*/ 53, /*->2659*/
/*  2606*/   OPC_RecordChild0, // #0 = $src2
/*  2607*/   OPC_RecordChild1, // #1 = $imm
/*  2608*/   OPC_Scope, 30, /*->2640*/ // 3 children in Scope
/*  2610*/    OPC_MoveChild1,
/*  2611*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2614*/    OPC_Scope, 11, /*->2627*/ // 2 children in Scope
/*  2616*/     OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  2617*/     OPC_MoveParent,
/*  2618*/     OPC_CheckTypeI64,
/*  2619*/     OPC_EmitConvertToTarget1,
/*  2620*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::OR_ri),
                   MVT::i64, 2/*#Ops*/, 0, 2, 
               // Src: (or:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
               // Dst: (OR_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  2627*/    /*Scope*/ 11, /*->2639*/
/*  2628*/     OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2629*/     OPC_MoveParent,
/*  2630*/     OPC_CheckTypeI32,
/*  2631*/     OPC_EmitConvertToTarget1,
/*  2632*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::OR_ri_32),
                   MVT::i32, 2/*#Ops*/, 0, 2, 
               // Src: (or:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
               // Dst: (OR_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  2639*/    0, /*End of Scope*/
/*  2640*/   /*Scope*/ 8, /*->2649*/
/*  2641*/    OPC_CheckTypeI64,
/*  2642*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::OR_rr),
                  MVT::i64, 2/*#Ops*/, 0, 1, 
              // Src: (or:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
              // Dst: (OR_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  2649*/   /*Scope*/ 8, /*->2658*/
/*  2650*/    OPC_CheckTypeI32,
/*  2651*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::OR_rr_32),
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (or:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
              // Dst: (OR_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  2658*/   0, /*End of Scope*/
/*  2659*/  0, /*End of Scope*/
/*  2660*/ /*SwitchOpcode*/ 21, TARGET_VAL(ISD::CALLSEQ_START),// ->2684
/*  2663*/  OPC_RecordNode, // #0 = 'BPFcallseq_start' chained node
/*  2664*/  OPC_RecordChild1, // #1 = $amt1
/*  2665*/  OPC_MoveChild1,
/*  2666*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/*  2669*/  OPC_MoveSibling2,
/*  2670*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/*  2673*/  OPC_RecordNode, // #2 = $amt2
/*  2674*/  OPC_MoveParent,
/*  2675*/  OPC_EmitMergeInputChains1_0,
/*  2676*/  OPC_MorphNodeTo1, TARGET_VAL(BPF::ADJCALLSTACKDOWN), 0|OPFL_Chain|OPFL_GlueOutput,
                MVT::i64, 2/*#Ops*/, 1, 2, 
            // Src: (BPFcallseq_start (timm:{ *:[iPTR] }):$amt1, (timm:{ *:[iPTR] }):$amt2) - Complexity = 9
            // Dst: (ADJCALLSTACKDOWN:{ *:[i64] } (timm:{ *:[i64] }):$amt1, (timm:{ *:[i64] }):$amt2)
/*  2684*/ /*SwitchOpcode*/ 22, TARGET_VAL(ISD::CALLSEQ_END),// ->2709
/*  2687*/  OPC_RecordNode, // #0 = 'BPFcallseq_end' chained node
/*  2688*/  OPC_CaptureGlueInput,
/*  2689*/  OPC_RecordChild1, // #1 = $amt1
/*  2690*/  OPC_MoveChild1,
/*  2691*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/*  2694*/  OPC_MoveSibling2,
/*  2695*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/*  2698*/  OPC_RecordNode, // #2 = $amt2
/*  2699*/  OPC_MoveParent,
/*  2700*/  OPC_EmitMergeInputChains1_0,
/*  2701*/  OPC_MorphNodeTo1, TARGET_VAL(BPF::ADJCALLSTACKUP), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                MVT::i64, 2/*#Ops*/, 1, 2, 
            // Src: (BPFcallseq_end (timm:{ *:[iPTR] }):$amt1, (timm:{ *:[iPTR] }):$amt2) - Complexity = 9
            // Dst: (ADJCALLSTACKUP:{ *:[i64] } (timm:{ *:[i64] }):$amt1, (timm:{ *:[i64] }):$amt2)
/*  2709*/ /*SwitchOpcode*/ 27, TARGET_VAL(BPFISD::MEMCPY),// ->2739
/*  2712*/  OPC_RecordNode, // #0 = 'BPFmemcpy' chained node
/*  2713*/  OPC_CaptureGlueInput,
/*  2714*/  OPC_RecordChild1, // #1 = $dst
/*  2715*/  OPC_RecordChild2, // #2 = $src
/*  2716*/  OPC_RecordChild3, // #3 = $len
/*  2717*/  OPC_MoveChild3,
/*  2718*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2721*/  OPC_MoveSibling4,
/*  2722*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2725*/  OPC_RecordNode, // #4 = $align
/*  2726*/  OPC_MoveParent,
/*  2727*/  OPC_EmitMergeInputChains1_0,
/*  2728*/  OPC_EmitConvertToTarget3,
/*  2729*/  OPC_EmitConvertToTarget4,
/*  2730*/  OPC_MorphNodeTo0, TARGET_VAL(BPF::MEMCPY), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                4/*#Ops*/, 1, 2, 5, 6, 
            // Src: (BPFmemcpy GPR:{ *:[i64] }:$dst, GPR:{ *:[i64] }:$src, (imm:{ *:[i64] }):$len, (imm:{ *:[i64] }):$align) - Complexity = 9
            // Dst: (MEMCPY GPR:{ *:[i64] }:$dst, GPR:{ *:[i64] }:$src, (imm:{ *:[i64] }):$len, (imm:{ *:[i64] }):$align)
/*  2739*/ /*SwitchOpcode*/ 78, TARGET_VAL(ISD::SUB),// ->2820
/*  2742*/  OPC_Scope, 21, /*->2765*/ // 2 children in Scope
/*  2744*/   OPC_CheckChild0Integer, 0, 
/*  2746*/   OPC_RecordChild1, // #0 = $src
/*  2747*/   OPC_SwitchType /*2 cases */, 6, MVT::i64,// ->2756
/*  2750*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::NEG_64),
                  MVT::i64, 1/*#Ops*/, 0, 
              // Src: (sub:{ *:[i64] } 0:{ *:[i64] }, i64:{ *:[i64] }:$src) - Complexity = 8
              // Dst: (NEG_64:{ *:[i64] } i64:{ *:[i64] }:$src)
/*  2756*/   /*SwitchType*/ 6, MVT::i32,// ->2764
/*  2758*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::NEG_32),
                  MVT::i32, 1/*#Ops*/, 0, 
              // Src: (sub:{ *:[i32] } 0:{ *:[i32] }, i32:{ *:[i32] }:$src) - Complexity = 8
              // Dst: (NEG_32:{ *:[i32] } i32:{ *:[i32] }:$src)
/*  2764*/   0, // EndSwitchType
/*  2765*/  /*Scope*/ 53, /*->2819*/
/*  2766*/   OPC_RecordChild0, // #0 = $src2
/*  2767*/   OPC_RecordChild1, // #1 = $imm
/*  2768*/   OPC_Scope, 30, /*->2800*/ // 3 children in Scope
/*  2770*/    OPC_MoveChild1,
/*  2771*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2774*/    OPC_Scope, 11, /*->2787*/ // 2 children in Scope
/*  2776*/     OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  2777*/     OPC_MoveParent,
/*  2778*/     OPC_CheckTypeI64,
/*  2779*/     OPC_EmitConvertToTarget1,
/*  2780*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::SUB_ri),
                   MVT::i64, 2/*#Ops*/, 0, 2, 
               // Src: (sub:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
               // Dst: (SUB_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  2787*/    /*Scope*/ 11, /*->2799*/
/*  2788*/     OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2789*/     OPC_MoveParent,
/*  2790*/     OPC_CheckTypeI32,
/*  2791*/     OPC_EmitConvertToTarget1,
/*  2792*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::SUB_ri_32),
                   MVT::i32, 2/*#Ops*/, 0, 2, 
               // Src: (sub:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
               // Dst: (SUB_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  2799*/    0, /*End of Scope*/
/*  2800*/   /*Scope*/ 8, /*->2809*/
/*  2801*/    OPC_CheckTypeI64,
/*  2802*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SUB_rr),
                  MVT::i64, 2/*#Ops*/, 0, 1, 
              // Src: (sub:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
              // Dst: (SUB_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  2809*/   /*Scope*/ 8, /*->2818*/
/*  2810*/    OPC_CheckTypeI32,
/*  2811*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SUB_rr_32),
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (sub:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
              // Dst: (SUB_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  2818*/   0, /*End of Scope*/
/*  2819*/  0, /*End of Scope*/
/*  2820*/ /*SwitchOpcode*/ 83, TARGET_VAL(ISD::AND),// ->2906
/*  2823*/  OPC_Scope, 26, /*->2851*/ // 2 children in Scope
/*  2825*/   OPC_CheckAndImm, 127|128,127|128,127|128,127|128,15/*4294967295*/, 
/*  2831*/   OPC_RecordChild0, // #0 = $src
/*  2832*/   OPC_CheckTypeI64,
/*  2833*/   OPC_EmitInteger64, 64, 
/*  2835*/   OPC_EmitNode1None, TARGET_VAL(BPF::SLL_ri),
                 MVT::i64, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2842*/   OPC_EmitInteger64, 64, 
/*  2844*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRL_ri),
                 MVT::i64, 2/*#Ops*/, 2, 3, 
             // Src: (and:{ *:[i64] } GPR:{ *:[i64] }:$src, 4294967295:{ *:[i64] }) - Complexity = 8
             // Dst: (SRL_ri:{ *:[i64] } (SLL_ri:{ *:[i64] } GPR:{ *:[i64] }:$src, 32:{ *:[i64] }), 32:{ *:[i64] })
/*  2851*/  /*Scope*/ 53, /*->2905*/
/*  2852*/   OPC_RecordChild0, // #0 = $src2
/*  2853*/   OPC_RecordChild1, // #1 = $imm
/*  2854*/   OPC_Scope, 30, /*->2886*/ // 3 children in Scope
/*  2856*/    OPC_MoveChild1,
/*  2857*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2860*/    OPC_Scope, 11, /*->2873*/ // 2 children in Scope
/*  2862*/     OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  2863*/     OPC_MoveParent,
/*  2864*/     OPC_CheckTypeI64,
/*  2865*/     OPC_EmitConvertToTarget1,
/*  2866*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::AND_ri),
                   MVT::i64, 2/*#Ops*/, 0, 2, 
               // Src: (and:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
               // Dst: (AND_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  2873*/    /*Scope*/ 11, /*->2885*/
/*  2874*/     OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2875*/     OPC_MoveParent,
/*  2876*/     OPC_CheckTypeI32,
/*  2877*/     OPC_EmitConvertToTarget1,
/*  2878*/     OPC_MorphNodeTo1None, TARGET_VAL(BPF::AND_ri_32),
                   MVT::i32, 2/*#Ops*/, 0, 2, 
               // Src: (and:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
               // Dst: (AND_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  2885*/    0, /*End of Scope*/
/*  2886*/   /*Scope*/ 8, /*->2895*/
/*  2887*/    OPC_CheckTypeI64,
/*  2888*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::AND_rr),
                  MVT::i64, 2/*#Ops*/, 0, 1, 
              // Src: (and:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
              // Dst: (AND_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  2895*/   /*Scope*/ 8, /*->2904*/
/*  2896*/    OPC_CheckTypeI32,
/*  2897*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::AND_rr_32),
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (and:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
              // Dst: (AND_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  2904*/   0, /*End of Scope*/
/*  2905*/  0, /*End of Scope*/
/*  2906*/ /*SwitchOpcode*/ 57, TARGET_VAL(ISD::SHL),// ->2966
/*  2909*/  OPC_RecordChild0, // #0 = $src2
/*  2910*/  OPC_RecordChild1, // #1 = $imm
/*  2911*/  OPC_Scope, 32, /*->2945*/ // 3 children in Scope
/*  2913*/   OPC_MoveChild1,
/*  2914*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2917*/   OPC_SwitchType /*2 cases */, 11, MVT::i64,// ->2931
/*  2920*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  2921*/    OPC_MoveParent,
/*  2922*/    OPC_CheckTypeI64,
/*  2923*/    OPC_EmitConvertToTarget1,
/*  2924*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SLL_ri),
                  MVT::i64, 2/*#Ops*/, 0, 2, 
              // Src: (shl:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
              // Dst: (SLL_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  2931*/   /*SwitchType*/ 11, MVT::i32,// ->2944
/*  2933*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2934*/    OPC_MoveParent,
/*  2935*/    OPC_CheckTypeI32,
/*  2936*/    OPC_EmitConvertToTarget1,
/*  2937*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SLL_ri_32),
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (shl:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
              // Dst: (SLL_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  2944*/   0, // EndSwitchType
/*  2945*/  /*Scope*/ 9, /*->2955*/
/*  2946*/   OPC_CheckChild1TypeI64,
/*  2947*/   OPC_CheckTypeI64,
/*  2948*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SLL_rr),
                 MVT::i64, 2/*#Ops*/, 0, 1, 
             // Src: (shl:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (SLL_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  2955*/  /*Scope*/ 9, /*->2965*/
/*  2956*/   OPC_CheckChild1TypeI32,
/*  2957*/   OPC_CheckTypeI32,
/*  2958*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SLL_rr_32),
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (shl:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (SLL_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  2965*/  0, /*End of Scope*/
/*  2966*/ /*SwitchOpcode*/ 53, TARGET_VAL(ISD::XOR),// ->3022
/*  2969*/  OPC_RecordChild0, // #0 = $src2
/*  2970*/  OPC_RecordChild1, // #1 = $imm
/*  2971*/  OPC_Scope, 30, /*->3003*/ // 3 children in Scope
/*  2973*/   OPC_MoveChild1,
/*  2974*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2977*/   OPC_Scope, 11, /*->2990*/ // 2 children in Scope
/*  2979*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  2980*/    OPC_MoveParent,
/*  2981*/    OPC_CheckTypeI64,
/*  2982*/    OPC_EmitConvertToTarget1,
/*  2983*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::XOR_ri),
                  MVT::i64, 2/*#Ops*/, 0, 2, 
              // Src: (xor:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
              // Dst: (XOR_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  2990*/   /*Scope*/ 11, /*->3002*/
/*  2991*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  2992*/    OPC_MoveParent,
/*  2993*/    OPC_CheckTypeI32,
/*  2994*/    OPC_EmitConvertToTarget1,
/*  2995*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::XOR_ri_32),
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (xor:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
              // Dst: (XOR_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  3002*/   0, /*End of Scope*/
/*  3003*/  /*Scope*/ 8, /*->3012*/
/*  3004*/   OPC_CheckTypeI64,
/*  3005*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::XOR_rr),
                 MVT::i64, 2/*#Ops*/, 0, 1, 
             // Src: (xor:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (XOR_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  3012*/  /*Scope*/ 8, /*->3021*/
/*  3013*/   OPC_CheckTypeI32,
/*  3014*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::XOR_rr_32),
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (xor:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (XOR_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  3021*/  0, /*End of Scope*/
/*  3022*/ /*SwitchOpcode*/ 57, TARGET_VAL(ISD::SRA),// ->3082
/*  3025*/  OPC_RecordChild0, // #0 = $src2
/*  3026*/  OPC_RecordChild1, // #1 = $imm
/*  3027*/  OPC_Scope, 32, /*->3061*/ // 3 children in Scope
/*  3029*/   OPC_MoveChild1,
/*  3030*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  3033*/   OPC_SwitchType /*2 cases */, 11, MVT::i64,// ->3047
/*  3036*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  3037*/    OPC_MoveParent,
/*  3038*/    OPC_CheckTypeI64,
/*  3039*/    OPC_EmitConvertToTarget1,
/*  3040*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRA_ri),
                  MVT::i64, 2/*#Ops*/, 0, 2, 
              // Src: (sra:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
              // Dst: (SRA_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  3047*/   /*SwitchType*/ 11, MVT::i32,// ->3060
/*  3049*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  3050*/    OPC_MoveParent,
/*  3051*/    OPC_CheckTypeI32,
/*  3052*/    OPC_EmitConvertToTarget1,
/*  3053*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRA_ri_32),
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (sra:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
              // Dst: (SRA_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  3060*/   0, // EndSwitchType
/*  3061*/  /*Scope*/ 9, /*->3071*/
/*  3062*/   OPC_CheckChild1TypeI64,
/*  3063*/   OPC_CheckTypeI64,
/*  3064*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRA_rr),
                 MVT::i64, 2/*#Ops*/, 0, 1, 
             // Src: (sra:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (SRA_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  3071*/  /*Scope*/ 9, /*->3081*/
/*  3072*/   OPC_CheckChild1TypeI32,
/*  3073*/   OPC_CheckTypeI32,
/*  3074*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRA_rr_32),
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (sra:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (SRA_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  3081*/  0, /*End of Scope*/
/*  3082*/ /*SwitchOpcode*/ 53, TARGET_VAL(ISD::MUL),// ->3138
/*  3085*/  OPC_RecordChild0, // #0 = $src2
/*  3086*/  OPC_RecordChild1, // #1 = $imm
/*  3087*/  OPC_Scope, 30, /*->3119*/ // 3 children in Scope
/*  3089*/   OPC_MoveChild1,
/*  3090*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  3093*/   OPC_Scope, 11, /*->3106*/ // 2 children in Scope
/*  3095*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  3096*/    OPC_MoveParent,
/*  3097*/    OPC_CheckTypeI64,
/*  3098*/    OPC_EmitConvertToTarget1,
/*  3099*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MUL_ri),
                  MVT::i64, 2/*#Ops*/, 0, 2, 
              // Src: (mul:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
              // Dst: (MUL_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  3106*/   /*Scope*/ 11, /*->3118*/
/*  3107*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  3108*/    OPC_MoveParent,
/*  3109*/    OPC_CheckTypeI32,
/*  3110*/    OPC_EmitConvertToTarget1,
/*  3111*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MUL_ri_32),
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (mul:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
              // Dst: (MUL_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  3118*/   0, /*End of Scope*/
/*  3119*/  /*Scope*/ 8, /*->3128*/
/*  3120*/   OPC_CheckTypeI64,
/*  3121*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::MUL_rr),
                 MVT::i64, 2/*#Ops*/, 0, 1, 
             // Src: (mul:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (MUL_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  3128*/  /*Scope*/ 8, /*->3137*/
/*  3129*/   OPC_CheckTypeI32,
/*  3130*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::MUL_rr_32),
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (mul:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (MUL_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  3137*/  0, /*End of Scope*/
/*  3138*/ /*SwitchOpcode*/ 53, TARGET_VAL(ISD::UDIV),// ->3194
/*  3141*/  OPC_RecordChild0, // #0 = $src2
/*  3142*/  OPC_RecordChild1, // #1 = $imm
/*  3143*/  OPC_Scope, 30, /*->3175*/ // 3 children in Scope
/*  3145*/   OPC_MoveChild1,
/*  3146*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  3149*/   OPC_Scope, 11, /*->3162*/ // 2 children in Scope
/*  3151*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  3152*/    OPC_MoveParent,
/*  3153*/    OPC_CheckTypeI64,
/*  3154*/    OPC_EmitConvertToTarget1,
/*  3155*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::DIV_ri),
                  MVT::i64, 2/*#Ops*/, 0, 2, 
              // Src: (udiv:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
              // Dst: (DIV_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  3162*/   /*Scope*/ 11, /*->3174*/
/*  3163*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  3164*/    OPC_MoveParent,
/*  3165*/    OPC_CheckTypeI32,
/*  3166*/    OPC_EmitConvertToTarget1,
/*  3167*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::DIV_ri_32),
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (udiv:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
              // Dst: (DIV_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  3174*/   0, /*End of Scope*/
/*  3175*/  /*Scope*/ 8, /*->3184*/
/*  3176*/   OPC_CheckTypeI64,
/*  3177*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::DIV_rr),
                 MVT::i64, 2/*#Ops*/, 0, 1, 
             // Src: (udiv:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (DIV_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  3184*/  /*Scope*/ 8, /*->3193*/
/*  3185*/   OPC_CheckTypeI32,
/*  3186*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::DIV_rr_32),
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (udiv:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (DIV_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  3193*/  0, /*End of Scope*/
/*  3194*/ /*SwitchOpcode*/ 53, TARGET_VAL(ISD::UREM),// ->3250
/*  3197*/  OPC_RecordChild0, // #0 = $src2
/*  3198*/  OPC_RecordChild1, // #1 = $imm
/*  3199*/  OPC_Scope, 30, /*->3231*/ // 3 children in Scope
/*  3201*/   OPC_MoveChild1,
/*  3202*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  3205*/   OPC_Scope, 11, /*->3218*/ // 2 children in Scope
/*  3207*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  3208*/    OPC_MoveParent,
/*  3209*/    OPC_CheckTypeI64,
/*  3210*/    OPC_EmitConvertToTarget1,
/*  3211*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOD_ri),
                  MVT::i64, 2/*#Ops*/, 0, 2, 
              // Src: (urem:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
              // Dst: (MOD_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  3218*/   /*Scope*/ 11, /*->3230*/
/*  3219*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  3220*/    OPC_MoveParent,
/*  3221*/    OPC_CheckTypeI32,
/*  3222*/    OPC_EmitConvertToTarget1,
/*  3223*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOD_ri_32),
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (urem:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
              // Dst: (MOD_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  3230*/   0, /*End of Scope*/
/*  3231*/  /*Scope*/ 8, /*->3240*/
/*  3232*/   OPC_CheckTypeI64,
/*  3233*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOD_rr),
                 MVT::i64, 2/*#Ops*/, 0, 1, 
             // Src: (urem:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (MOD_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  3240*/  /*Scope*/ 8, /*->3249*/
/*  3241*/   OPC_CheckTypeI32,
/*  3242*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOD_rr_32),
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (urem:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (MOD_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  3249*/  0, /*End of Scope*/
/*  3250*/ /*SwitchOpcode*/ 57, TARGET_VAL(ISD::SDIV),// ->3310
/*  3253*/  OPC_RecordChild0, // #0 = $src2
/*  3254*/  OPC_RecordChild1, // #1 = $imm
/*  3255*/  OPC_Scope, 32, /*->3289*/ // 3 children in Scope
/*  3257*/   OPC_MoveChild1,
/*  3258*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  3261*/   OPC_Scope, 12, /*->3275*/ // 2 children in Scope
/*  3263*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  3264*/    OPC_MoveParent,
/*  3265*/    OPC_CheckTypeI64,
/*  3266*/    OPC_CheckPatternPredicate2, // (Subtarget->hasSdivSmod())
/*  3267*/    OPC_EmitConvertToTarget1,
/*  3268*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SDIV_ri),
                  MVT::i64, 2/*#Ops*/, 0, 2, 
              // Src: (sdiv:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
              // Dst: (SDIV_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  3275*/   /*Scope*/ 12, /*->3288*/
/*  3276*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  3277*/    OPC_MoveParent,
/*  3278*/    OPC_CheckTypeI32,
/*  3279*/    OPC_CheckPatternPredicate2, // (Subtarget->hasSdivSmod())
/*  3280*/    OPC_EmitConvertToTarget1,
/*  3281*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SDIV_ri_32),
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (sdiv:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
              // Dst: (SDIV_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  3288*/   0, /*End of Scope*/
/*  3289*/  /*Scope*/ 9, /*->3299*/
/*  3290*/   OPC_CheckTypeI64,
/*  3291*/   OPC_CheckPatternPredicate2, // (Subtarget->hasSdivSmod())
/*  3292*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SDIV_rr),
                 MVT::i64, 2/*#Ops*/, 0, 1, 
             // Src: (sdiv:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (SDIV_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  3299*/  /*Scope*/ 9, /*->3309*/
/*  3300*/   OPC_CheckTypeI32,
/*  3301*/   OPC_CheckPatternPredicate2, // (Subtarget->hasSdivSmod())
/*  3302*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SDIV_rr_32),
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (sdiv:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (SDIV_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  3309*/  0, /*End of Scope*/
/*  3310*/ /*SwitchOpcode*/ 57, TARGET_VAL(ISD::SREM),// ->3370
/*  3313*/  OPC_RecordChild0, // #0 = $src2
/*  3314*/  OPC_RecordChild1, // #1 = $imm
/*  3315*/  OPC_Scope, 32, /*->3349*/ // 3 children in Scope
/*  3317*/   OPC_MoveChild1,
/*  3318*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  3321*/   OPC_Scope, 12, /*->3335*/ // 2 children in Scope
/*  3323*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  3324*/    OPC_MoveParent,
/*  3325*/    OPC_CheckTypeI64,
/*  3326*/    OPC_CheckPatternPredicate2, // (Subtarget->hasSdivSmod())
/*  3327*/    OPC_EmitConvertToTarget1,
/*  3328*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SMOD_ri),
                  MVT::i64, 2/*#Ops*/, 0, 2, 
              // Src: (srem:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm) - Complexity = 7
              // Dst: (SMOD_ri:{ *:[i64] } GPR:{ *:[i64] }:$src2, (imm:{ *:[i64] }):$imm)
/*  3335*/   /*Scope*/ 12, /*->3348*/
/*  3336*/    OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  3337*/    OPC_MoveParent,
/*  3338*/    OPC_CheckTypeI32,
/*  3339*/    OPC_CheckPatternPredicate2, // (Subtarget->hasSdivSmod())
/*  3340*/    OPC_EmitConvertToTarget1,
/*  3341*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::SMOD_ri_32),
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (srem:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm) - Complexity = 7
              // Dst: (SMOD_ri_32:{ *:[i32] } GPR32:{ *:[i32] }:$src2, (imm:{ *:[i32] }):$imm)
/*  3348*/   0, /*End of Scope*/
/*  3349*/  /*Scope*/ 9, /*->3359*/
/*  3350*/   OPC_CheckTypeI64,
/*  3351*/   OPC_CheckPatternPredicate2, // (Subtarget->hasSdivSmod())
/*  3352*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SMOD_rr),
                 MVT::i64, 2/*#Ops*/, 0, 1, 
             // Src: (srem:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (SMOD_rr:{ *:[i64] } i64:{ *:[i64] }:$src2, i64:{ *:[i64] }:$src)
/*  3359*/  /*Scope*/ 9, /*->3369*/
/*  3360*/   OPC_CheckTypeI32,
/*  3361*/   OPC_CheckPatternPredicate2, // (Subtarget->hasSdivSmod())
/*  3362*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SMOD_rr_32),
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (srem:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (SMOD_rr_32:{ *:[i32] } i32:{ *:[i32] }:$src2, i32:{ *:[i32] }:$src)
/*  3369*/  0, /*End of Scope*/
/*  3370*/ /*SwitchOpcode*/ 13, TARGET_VAL(BPFISD::Wrapper),// ->3386
/*  3373*/  OPC_RecordChild0, // #0 = $in
/*  3374*/  OPC_MoveChild0,
/*  3375*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetGlobalAddress),
/*  3378*/  OPC_MoveParent,
/*  3379*/  OPC_CheckTypeI64,
/*  3380*/  OPC_MorphNodeTo1None, TARGET_VAL(BPF::LD_imm64),
                MVT::i64, 1/*#Ops*/, 0, 
            // Src: (BPFWrapper:{ *:[i64] } (tglobaladdr:{ *:[i64] }):$in) - Complexity = 6
            // Dst: (LD_imm64:{ *:[i64] } (tglobaladdr:{ *:[i64] }):$in)
/*  3386*/ /*SwitchOpcode*/ 56, TARGET_VAL(BPFISD::CALL),// ->3445
/*  3389*/  OPC_RecordNode, // #0 = 'BPFcall' chained node
/*  3390*/  OPC_CaptureGlueInput,
/*  3391*/  OPC_RecordChild1, // #1 = $dst
/*  3392*/  OPC_Scope, 40, /*->3434*/ // 2 children in Scope
/*  3394*/   OPC_MoveChild1,
/*  3395*/   OPC_SwitchOpcode /*3 cases */, 9, TARGET_VAL(ISD::TargetGlobalAddress),// ->3408
/*  3399*/    OPC_MoveParent,
/*  3400*/    OPC_EmitMergeInputChains1_0,
/*  3401*/    OPC_MorphNodeTo1, TARGET_VAL(BPF::JAL), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                  MVT::i64, 1/*#Ops*/, 1, 
              // Src: (BPFcall (tglobaladdr:{ *:[iPTR] }):$dst) - Complexity = 6
              // Dst: (JAL:{ *:[i64] } (tglobaladdr:{ *:[i64] }):$dst)
/*  3408*/   /*SwitchOpcode*/ 9, TARGET_VAL(ISD::TargetExternalSymbol),// ->3420
/*  3411*/    OPC_MoveParent,
/*  3412*/    OPC_EmitMergeInputChains1_0,
/*  3413*/    OPC_MorphNodeTo1, TARGET_VAL(BPF::JAL), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                  MVT::i64, 1/*#Ops*/, 1, 
              // Src: (BPFcall (texternalsym:{ *:[iPTR] }):$dst) - Complexity = 6
              // Dst: (JAL:{ *:[i64] } (texternalsym:{ *:[i64] }):$dst)
/*  3420*/   /*SwitchOpcode*/ 10, TARGET_VAL(ISD::Constant),// ->3433
/*  3423*/    OPC_MoveParent,
/*  3424*/    OPC_EmitMergeInputChains1_0,
/*  3425*/    OPC_EmitConvertToTarget1,
/*  3426*/    OPC_MorphNodeTo1, TARGET_VAL(BPF::JAL), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                  MVT::i64, 1/*#Ops*/, 2, 
              // Src: (BPFcall (imm:{ *:[iPTR] }):$dst) - Complexity = 6
              // Dst: (JAL:{ *:[i64] } (imm:{ *:[i64] }):$dst)
/*  3433*/   0, // EndSwitchOpcode
/*  3434*/  /*Scope*/ 9, /*->3444*/
/*  3435*/   OPC_CheckChild1TypeI64,
/*  3436*/   OPC_EmitMergeInputChains1_0,
/*  3437*/   OPC_MorphNodeTo1, TARGET_VAL(BPF::JALX), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                 MVT::i64, 1/*#Ops*/, 1, 
             // Src: (BPFcall GPR:{ *:[i64] }:$dst) - Complexity = 3
             // Dst: (JALX:{ *:[i64] } GPR:{ *:[i64] }:$dst)
/*  3444*/  0, /*End of Scope*/
/*  3445*/ /*SwitchOpcode*/ 34, TARGET_VAL(ISD::Constant),// ->3482
/*  3448*/  OPC_RecordNode, // #0 = $imm
/*  3449*/  OPC_SwitchType /*2 cases */, 19, MVT::i64,// ->3471
/*  3452*/   OPC_Scope, 8, /*->3462*/ // 2 children in Scope
/*  3454*/    OPC_CheckPredicate0,  // Predicate_i64immSExt32
/*  3455*/    OPC_EmitConvertToTarget0,
/*  3456*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOV_ri),
                  MVT::i64, 1/*#Ops*/, 1, 
              // Src: (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$imm - Complexity = 4
              // Dst: (MOV_ri:{ *:[i64] } (imm:{ *:[i64] }):$imm)
/*  3462*/   /*Scope*/ 7, /*->3470*/
/*  3463*/    OPC_EmitConvertToTarget0,
/*  3464*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::LD_imm64),
                  MVT::i64, 1/*#Ops*/, 1, 
              // Src: (imm:{ *:[i64] }):$imm - Complexity = 3
              // Dst: (LD_imm64:{ *:[i64] } (imm:{ *:[i64] }):$imm)
/*  3470*/   0, /*End of Scope*/
/*  3471*/  /*SwitchType*/ 8, MVT::i32,// ->3481
/*  3473*/   OPC_CheckPredicate0,  // Predicate_i32immSExt32
/*  3474*/   OPC_EmitConvertToTarget0,
/*  3475*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOV_ri_32),
                 MVT::i32, 1/*#Ops*/, 1, 
             // Src: (imm:{ *:[i32] })<<P:Predicate_i32immSExt32>>:$imm - Complexity = 4
             // Dst: (MOV_ri_32:{ *:[i32] } (imm:{ *:[i32] }):$imm)
/*  3481*/  0, // EndSwitchType
/*  3482*/ /*SwitchOpcode*/ 63, TARGET_VAL(ISD::SIGN_EXTEND_INREG),// ->3548
/*  3485*/  OPC_RecordChild0, // #0 = $src
/*  3486*/  OPC_MoveChild1,
/*  3487*/  OPC_Scope, 23, /*->3512*/ // 3 children in Scope
/*  3489*/   OPC_CheckValueType, MVT::i8,
/*  3491*/   OPC_MoveParent,
/*  3492*/   OPC_SwitchType /*2 cases */, 7, MVT::i64,// ->3502
/*  3495*/    OPC_CheckPatternPredicate3, // (Subtarget->hasMovsx())
/*  3496*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOVSX_rr_8),
                  MVT::i64, 1/*#Ops*/, 0, 
              // Src: (sext_inreg:{ *:[i64] } GPR:{ *:[i64] }:$src, i8:{ *:[Other] }) - Complexity = 3
              // Dst: (MOVSX_rr_8:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  3502*/   /*SwitchType*/ 7, MVT::i32,// ->3511
/*  3504*/    OPC_CheckPatternPredicate3, // (Subtarget->hasMovsx())
/*  3505*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOVSX_rr_32_8),
                  MVT::i32, 1/*#Ops*/, 0, 
              // Src: (sext_inreg:{ *:[i32] } GPR32:{ *:[i32] }:$src, i8:{ *:[Other] }) - Complexity = 3
              // Dst: (MOVSX_rr_32_8:{ *:[i32] } GPR32:{ *:[i32] }:$src)
/*  3511*/   0, // EndSwitchType
/*  3512*/  /*Scope*/ 23, /*->3536*/
/*  3513*/   OPC_CheckValueType, MVT::i16,
/*  3515*/   OPC_MoveParent,
/*  3516*/   OPC_SwitchType /*2 cases */, 7, MVT::i64,// ->3526
/*  3519*/    OPC_CheckPatternPredicate3, // (Subtarget->hasMovsx())
/*  3520*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOVSX_rr_16),
                  MVT::i64, 1/*#Ops*/, 0, 
              // Src: (sext_inreg:{ *:[i64] } GPR:{ *:[i64] }:$src, i16:{ *:[Other] }) - Complexity = 3
              // Dst: (MOVSX_rr_16:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  3526*/   /*SwitchType*/ 7, MVT::i32,// ->3535
/*  3528*/    OPC_CheckPatternPredicate3, // (Subtarget->hasMovsx())
/*  3529*/    OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOVSX_rr_32_16),
                  MVT::i32, 1/*#Ops*/, 0, 
              // Src: (sext_inreg:{ *:[i32] } GPR32:{ *:[i32] }:$src, i16:{ *:[Other] }) - Complexity = 3
              // Dst: (MOVSX_rr_32_16:{ *:[i32] } GPR32:{ *:[i32] }:$src)
/*  3535*/   0, // EndSwitchType
/*  3536*/  /*Scope*/ 10, /*->3547*/
/*  3537*/   OPC_CheckValueType, MVT::i32,
/*  3539*/   OPC_MoveParent,
/*  3540*/   OPC_CheckPatternPredicate3, // (Subtarget->hasMovsx())
/*  3541*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOVSX_rr_32),
                 MVT::i64, 1/*#Ops*/, 0, 
             // Src: (sext_inreg:{ *:[i64] } GPR:{ *:[i64] }:$src, i32:{ *:[Other] }) - Complexity = 3
             // Dst: (MOVSX_rr_32:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  3547*/  0, /*End of Scope*/
/*  3548*/ /*SwitchOpcode*/ 13, TARGET_VAL(ISD::BR),// ->3564
/*  3551*/  OPC_RecordNode, // #0 = 'br' chained node
/*  3552*/  OPC_RecordChild1, // #1 = $BrDst
/*  3553*/  OPC_MoveChild1,
/*  3554*/  OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  3557*/  OPC_MoveParent,
/*  3558*/  OPC_EmitMergeInputChains1_0,
/*  3559*/  OPC_MorphNodeTo0Chain, TARGET_VAL(BPF::JMP),
                1/*#Ops*/, 1, 
            // Src: (br (bb:{ *:[Other] }):$BrDst) - Complexity = 3
            // Dst: (JMP (bb:{ *:[Other] }):$BrDst)
/*  3564*/ /*SwitchOpcode*/ 8, TARGET_VAL(BPFISD::RET_GLUE),// ->3575
/*  3567*/  OPC_RecordNode, // #0 = 'BPFretglue' chained node
/*  3568*/  OPC_CaptureGlueInput,
/*  3569*/  OPC_EmitMergeInputChains1_0,
/*  3570*/  OPC_MorphNodeTo0, TARGET_VAL(BPF::RET), 0|OPFL_Chain|OPFL_GlueInput|OPFL_Variadic0,
                0/*#Ops*/, 
            // Src: (BPFretglue) - Complexity = 3
            // Dst: (RET)
/*  3575*/ /*SwitchOpcode*/ 30, TARGET_VAL(ISD::BSWAP),// ->3608
/*  3578*/  OPC_RecordChild0, // #0 = $src
/*  3579*/  OPC_CheckTypeI64,
/*  3580*/  OPC_Scope, 7, /*->3589*/ // 3 children in Scope
/*  3582*/   OPC_CheckPatternPredicate7, // (Subtarget->hasBswap())
/*  3583*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::BSWAP64),
                 MVT::i64, 1/*#Ops*/, 0, 
             // Src: (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (BSWAP64:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  3589*/  /*Scope*/ 8, /*->3598*/
/*  3590*/   OPC_CheckPatternPredicate, 8, // (Subtarget->isLittleEndian())
/*  3592*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::BE64),
                 MVT::i64, 1/*#Ops*/, 0, 
             // Src: (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (BE64:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  3598*/  /*Scope*/ 8, /*->3607*/
/*  3599*/   OPC_CheckPatternPredicate, 9, // (!Subtarget->isLittleEndian())
/*  3601*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::LE64),
                 MVT::i64, 1/*#Ops*/, 0, 
             // Src: (bswap:{ *:[i64] } GPR:{ *:[i64] }:$src) - Complexity = 3
             // Dst: (LE64:{ *:[i64] } GPR:{ *:[i64] }:$src)
/*  3607*/  0, /*End of Scope*/
/*  3608*/ /*SwitchOpcode*/ 7, TARGET_VAL(ISD::ZERO_EXTEND),// ->3618
/*  3611*/  OPC_RecordChild0, // #0 = $src
/*  3612*/  OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOV_32_64),
                MVT::i64, 1/*#Ops*/, 0, 
            // Src: (zext:{ *:[i64] } GPR32:{ *:[i32] }:$src) - Complexity = 3
            // Dst: (MOV_32_64:{ *:[i64] } GPR32:{ *:[i32] }:$src)
/*  3618*/ /*SwitchOpcode*/ 10, TARGET_VAL(ISD::TRUNCATE),// ->3631
/*  3621*/  OPC_RecordChild0, // #0 = $src
/*  3622*/  OPC_EmitStringInteger32, BPF::sub_32,
/*  3624*/  OPC_MorphNodeTo1None, TARGET_VAL(TargetOpcode::EXTRACT_SUBREG),
                MVT::i32, 2/*#Ops*/, 0, 1, 
            // Src: (trunc:{ *:[i32] } GPR:{ *:[i64] }:$src) - Complexity = 3
            // Dst: (EXTRACT_SUBREG:{ *:[i32] } GPR:{ *:[i64] }:$src, sub_32:{ *:[i32] })
/*  3631*/ /*SwitchOpcode*/ 44, TARGET_VAL(ISD::SIGN_EXTEND),// ->3678
/*  3634*/  OPC_RecordChild0, // #0 = $src
/*  3635*/  OPC_Scope, 13, /*->3650*/ // 2 children in Scope
/*  3637*/   OPC_CheckPatternPredicate3, // (Subtarget->hasMovsx())
/*  3638*/   OPC_EmitNode1None, TARGET_VAL(BPF::MOV_32_64),
                 MVT::i64, 1/*#Ops*/, 0,  // Results = #1
/*  3644*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::MOVSX_rr_32),
                 MVT::i64, 1/*#Ops*/, 1, 
             // Src: (sext:{ *:[i64] } GPR32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (MOVSX_rr_32:{ *:[i64] } (MOV_32_64:{ *:[i64] } GPR32:{ *:[i32] }:$src))
/*  3650*/  /*Scope*/ 26, /*->3677*/
/*  3651*/   OPC_CheckPatternPredicate, 10, // (!Subtarget->hasMovsx())
/*  3653*/   OPC_EmitNode1None, TARGET_VAL(BPF::MOV_32_64),
                 MVT::i64, 1/*#Ops*/, 0,  // Results = #1
/*  3659*/   OPC_EmitInteger64, 64, 
/*  3661*/   OPC_EmitNode1None, TARGET_VAL(BPF::SLL_ri),
                 MVT::i64, 2/*#Ops*/, 1, 2,  // Results = #3
/*  3668*/   OPC_EmitInteger64, 64, 
/*  3670*/   OPC_MorphNodeTo1None, TARGET_VAL(BPF::SRA_ri),
                 MVT::i64, 2/*#Ops*/, 3, 4, 
             // Src: (sext:{ *:[i64] } GPR32:{ *:[i32] }:$src) - Complexity = 3
             // Dst: (SRA_ri:{ *:[i64] } (SLL_ri:{ *:[i64] } (MOV_32_64:{ *:[i64] } GPR32:{ *:[i32] }:$src), 32:{ *:[i64] }), 32:{ *:[i64] })
/*  3677*/  0, /*End of Scope*/
/*  3678*/ /*SwitchOpcode*/ 16, TARGET_VAL(ISD::ANY_EXTEND),// ->3697
/*  3681*/  OPC_RecordChild0, // #0 = $src
/*  3682*/  OPC_EmitNode1None, TARGET_VAL(TargetOpcode::IMPLICIT_DEF),
                MVT::i64, 0/*#Ops*/,  // Results = #1
/*  3687*/  OPC_EmitStringInteger32, BPF::sub_32,
/*  3689*/  OPC_MorphNodeTo1None, TARGET_VAL(TargetOpcode::INSERT_SUBREG),
                MVT::i64, 3/*#Ops*/, 1, 0, 2, 
            // Src: (anyext:{ *:[i64] } GPR32:{ *:[i32] }:$src) - Complexity = 3
            // Dst: (INSERT_SUBREG:{ *:[i64] } (IMPLICIT_DEF:{ *:[i64] }), GPR32:{ *:[i32] }:$src, sub_32:{ *:[i32] })
/*  3697*/ 0, // EndSwitchOpcode
    0
  }; // Total Array size is 3699 bytes

  // Opcode Histogram:
  // #OPC_Scope                                = 71
  // #OPC_RecordNode                           = 70
  // #OPC_RecordChild                          = 88
  // #OPC_RecordMemRef                         = 9
  // #OPC_CaptureGlueInput                     = 6
  // #OPC_MoveChild                            = 38
  // #OPC_MoveSibling                          = 54
  // #OPC_MoveParent                           = 93
  // #OPC_CheckSame                            = 0
  // #OPC_CheckChildSame                       = 0
  // #OPC_CheckPatternPredicate                = 69
  // #OPC_CheckPredicate                       = 152
  // #OPC_CheckOpcode                          = 90
  // #OPC_SwitchOpcode                         = 2
  // #OPC_CheckType                            = 67
  // #OPC_SwitchType                           = 20
  // #OPC_CheckChildType                       = 31
  // #OPC_CheckInteger                         = 0
  // #OPC_CheckChildInteger                    = 7
  // #OPC_CheckCondCode                        = 0
  // #OPC_CheckChild2CondCode                  = 0
  // #OPC_CheckValueType                       = 3
  // #OPC_CheckComplexPat                      = 57
  // #OPC_CheckAndImm                          = 1
  // #OPC_CheckOrImm                           = 0
  // #OPC_CheckImmAllOnesV                     = 0
  // #OPC_CheckImmAllZerosV                    = 0
  // #OPC_CheckFoldableChainNode               = 0
  // #OPC_EmitInteger                          = 10
  // #OPC_EmitStringInteger                    = 13
  // #OPC_EmitRegister                         = 0
  // #OPC_EmitConvertToTarget                  = 77
  // #OPC_EmitMergeInputChains                 = 115
  // #OPC_EmitCopyToReg                        = 2
  // #OPC_EmitNode                             = 18
  // #OPC_EmitNodeXForm                        = 3
  // #OPC_CompleteMatch                        = 0
  // #OPC_MorphNodeTo                          = 203

  #undef TARGET_VAL
  SelectCodeCommon(N, MatcherTable,sizeof(MatcherTable));
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckPatternPredicate(unsigned PredNo) const override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckPatternPredicate(unsigned PredNo) const
#if DAGISEL_INLINE
  override
#endif
{
  switch (PredNo) {
  default: llvm_unreachable("Invalid predicate in table?");
  case 0: return (Subtarget->getHasAlu32());
  case 1: return (!Subtarget->getHasAlu32());
  case 2: return (Subtarget->hasSdivSmod());
  case 3: return (Subtarget->hasMovsx());
  case 4: return (Subtarget->hasLdsx());
  case 5: return (Subtarget->hasStoreImm());
  case 6: return (Subtarget->getHasAlu32()) && (Subtarget->hasStoreImm());
  case 7: return (Subtarget->hasBswap());
  case 8: return (Subtarget->isLittleEndian());
  case 9: return (!Subtarget->isLittleEndian());
  case 10: return (!Subtarget->hasMovsx());
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckNodePredicate(SDNode *Node, unsigned PredNo) const override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckNodePredicate(SDNode *Node, unsigned PredNo) const
#if DAGISEL_INLINE
  override
#endif
{
  switch (PredNo) {
  default: llvm_unreachable("Invalid predicate in table?");
  case 0: {
    // Predicate_i32immSExt32
    // Predicate_i64immSExt32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return isInt<32>(N->getSExtValue()); 
  }
  case 1: {
    // Predicate_atomic_cmp_swap_32
    // Predicate_atomic_load_add_32
    // Predicate_atomic_load_and_32
    // Predicate_atomic_load_or_32
    // Predicate_atomic_load_sub_32
    // Predicate_atomic_load_xor_32
    // Predicate_atomic_swap_32
    // Predicate_extloadi32
    // Predicate_sextloadi32
    // Predicate_zextloadi32
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i32) return false;
return true;

  }
  case 2: {
    // Predicate_extloadi16
    // Predicate_sextloadi16
    // Predicate_zextloadi16
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i16) return false;
return true;

  }
  case 3: {
    // Predicate_extloadi8
    // Predicate_sextloadi8
    // Predicate_zextloadi8
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i8) return false;
return true;

  }
  case 4: {
    // Predicate_atomic_cmp_swap_64
    // Predicate_atomic_load_add_64
    // Predicate_atomic_load_and_64
    // Predicate_atomic_load_or_64
    // Predicate_atomic_load_sub_64
    // Predicate_atomic_load_xor_64
    // Predicate_atomic_swap_64
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i64) return false;
return true;

  }
  case 5: {
    // Predicate_unindexedstore
    SDNode *N = Node;
    (void)N;
if (cast<StoreSDNode>(N)->getAddressingMode() != ISD::UNINDEXED) return false;
return true;

  }
  case 6: {
    // Predicate_truncstore
    SDNode *N = Node;
    (void)N;
 if (!cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 7: {
    // Predicate_truncstorei16
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i16) return false;
 if (!cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 8: {
    // Predicate_truncstorei8
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i8) return false;
 if (!cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 9: {
    // Predicate_store
    SDNode *N = Node;
    (void)N;
 if (cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 10: {
    // Predicate_BPF_CC_EQ
    // Predicate_BPF_CC_EQ_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETEQ);
  }
  case 11: {
    // Predicate_BPF_CC_GTU
    // Predicate_BPF_CC_GTU_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETUGT);
  }
  case 12: {
    // Predicate_BPF_CC_GEU
    // Predicate_BPF_CC_GEU_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETUGE);
  }
  case 13: {
    // Predicate_BPF_CC_NE
    // Predicate_BPF_CC_NE_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETNE);
  }
  case 14: {
    // Predicate_BPF_CC_GT
    // Predicate_BPF_CC_GT_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETGT);
  }
  case 15: {
    // Predicate_BPF_CC_GE
    // Predicate_BPF_CC_GE_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETGE);
  }
  case 16: {
    // Predicate_BPF_CC_LTU
    // Predicate_BPF_CC_LTU_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETULT);
  }
  case 17: {
    // Predicate_BPF_CC_LEU
    // Predicate_BPF_CC_LEU_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETULE);
  }
  case 18: {
    // Predicate_BPF_CC_LT
    // Predicate_BPF_CC_LT_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETLT);
  }
  case 19: {
    // Predicate_BPF_CC_LE
    // Predicate_BPF_CC_LE_32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return (N->getZExtValue() == ISD::SETLE);
  }
  case 20: {
    // Predicate_truncstorei32
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i32) return false;
 if (!cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 21: {
    // Predicate_zextload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::ZEXTLOAD) return false;
return true;

  }
  case 22: {
    // Predicate_extload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::EXTLOAD) return false;
return true;

  }
  case 23: {
    // Predicate_sextload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::SEXTLOAD) return false;
return true;

  }
  case 24: {
    // Predicate_load
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::NON_EXTLOAD) return false;
return true;

  }
  case 25: {
    // Predicate_i64immZExt32
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
return isUInt<32>(N->getZExtValue()); 
  }
  case 26: {
    // Predicate_unindexedload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getAddressingMode() != ISD::UNINDEXED) return false;
return true;

  }
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckComplexPattern(SDNode *Root, SDNode *Parent,
      SDValue N, unsigned PatternNo,
      SmallVectorImpl<std::pair<SDValue, SDNode *>> &Result) override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckComplexPattern(SDNode *Root, SDNode *Parent,
      SDValue N, unsigned PatternNo,
      SmallVectorImpl<std::pair<SDValue, SDNode *>> &Result)
#if DAGISEL_INLINE
  override
#endif
{
  unsigned NextRes = Result.size();
  switch (PatternNo) {
  default: llvm_unreachable("Invalid pattern # in table?");
  case 0:
    Result.resize(NextRes+2);
  return SelectAddr(N, Result[NextRes+0].first, Result[NextRes+1].first);
  case 1:
    Result.resize(NextRes+2);
  return SelectFIAddr(N, Result[NextRes+0].first, Result[NextRes+1].first);
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
SDValue RunSDNodeXForm(SDValue V, unsigned XFormNo) override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
SDValue DAGISEL_CLASS_COLONCOLON RunSDNodeXForm(SDValue V, unsigned XFormNo)
#if DAGISEL_INLINE
  override
#endif
{
  switch (XFormNo) {
  default: llvm_unreachable("Invalid xform # in table?");
  case 0: {  // imm_to_i64
    ConstantSDNode *N = cast<ConstantSDNode>(V.getNode());

  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i64);

  }
  }
}
#endif // GET_DAGISEL_BODY


#ifdef DAGISEL_INLINE
#undef DAGISEL_INLINE
#endif
#ifdef DAGISEL_CLASS_COLONCOLON
#undef DAGISEL_CLASS_COLONCOLON
#endif
#ifdef GET_DAGISEL_DECL
#undef GET_DAGISEL_DECL
#endif
#ifdef GET_DAGISEL_BODY
#undef GET_DAGISEL_BODY
#endif
