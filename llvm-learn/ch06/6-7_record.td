------------- Classes -----------------
class Instr<bits<4> Instr:op = { ?, ?, ?, ? }, string Instr:desc = ?> {
  bits<4> opcode = { Instr:op{3}, Instr:op{2}, Instr:op{1}, Instr:op{0} };
  string name = Instr:desc;
}
------------- Defs -----------------
def MyBackend_rm {	// Instr
  bits<4> opcode = { 0, 0, 0, 0 };
  string name = "rm";
}
def MyBackend_rr {	// Instr
  bits<4> opcode = { 1, 1, 1, 1 };
  string name = "rr";
}
