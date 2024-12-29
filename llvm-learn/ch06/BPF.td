// See llvm-project/llvm/lib/Target/BPF/*.td  for details 

class InstructionEncoding {
  int Size;

  // ...
}

class Instruction : InstructionEncoding {
  string Namespace = "";
  string AsmString = "";

  // ...
}

class InstBPF<dag outs, dag ins, string asmstr, list<dag> pattern>
  : Instruction {
  field bits<64> Inst;
 
  // ...
}

class TYPE_ALU_JMP<bits<4> op, bits<1> srctype,
                   dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstBPF<outs, ins, asmstr, pattern> {

  let Inst{63-60} = op;
  let Inst{59} = srctype;
}

class ALU_RI<BPFOpClass Class, BPFArithOp Opc,
             dag outs, dag ins, string asmstr, list<dag> pattern>
    : TYPE_ALU_JMP<Opc.Value, BPF_K.Value, outs, ins, asmstr, pattern> {
  bits<4> dst;
  bits<32> imm;

  let Inst{51-48} = dst;
  let Inst{31-0} = imm;
  let BPFClass = Class;
}

class ALU_RR<BPFOpClass Class, BPFArithOp Opc,
             dag outs, dag ins, string asmstr, list<dag> pattern>
    : TYPE_ALU_JMP<Opc.Value, BPF_X.Value, outs, ins, asmstr, pattern> {
  // ...
}

multiclass ALU<BPFArithOp Opc, string OpcodeStr, SDNode OpNode> {
  def _rr : ALU_RR<BPF_ALU64, Opc,
                   (outs GPR:$dst),
                   (ins GPR:$src2, GPR:$src),
                   "$dst "#OpcodeStr#" $src",
                   [(set GPR:$dst, (OpNode i64:$src2, i64:$src))]>;
  def _ri : ALU_RI<BPF_ALU64, Opc,
                   (outs GPR:$dst),
                   (ins GPR:$src2, i64imm:$imm),
                   "$dst "#OpcodeStr#" $imm",
                   [(set GPR:$dst, (OpNode GPR:$src2, i64immSExt32:$imm))]>;
  def _rr_32 : ALU_RR<BPF_ALU, Opc,
                   (outs GPR32:$dst),
                   (ins GPR32:$src2, GPR32:$src),
                   "$dst "#OpcodeStr#" $src",
                   [(set GPR32:$dst, (OpNode i32:$src2, i32:$src))]>;
  def _ri_32 : ALU_RI<BPF_ALU, Opc,
                   (outs GPR32:$dst),
                   (ins GPR32:$src2, i32imm:$imm),
                   "$dst "#OpcodeStr#" $imm",
                   [(set GPR32:$dst, (OpNode GPR32:$src2, i32immSExt32:$imm))]>;
}

let Constraints = "$dst = $src2" in {
    let isAsCheapAsAMove = 1 in {
        defm ADD : ALU<BPF_ADD, "+=", add>;
        
        // ...
    }

    // ...
}

// generate record with:
// ../../opensource/llvm-project/build/bin/llvm-tblgen -I ../../opensource/llvm-project/llvm/include/ -I ../../opensource/llvm-project/llvm/lib/Target/BPF/ --print-records ../../opensource/llvm-project/llvm/lib/Target/BPF/BPF.td > BPF_record.td
