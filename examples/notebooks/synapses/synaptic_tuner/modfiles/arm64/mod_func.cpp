#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _AMPA_NMDA_STP_reg(void);
extern void _GABA_A_STP_reg(void);
extern void _Gfluct_reg(void);
extern void _cadad_reg(void);
extern void _cal2_reg(void);
extern void _can_mig_reg(void);
extern void _exp2syn_stp_reg(void);
extern void _gap_reg(void);
extern void _h_kole_reg(void);
extern void _imCA3_reg(void);
extern void _kBK_reg(void);
extern void _kap_BS_reg(void);
extern void _kdmc_BS_reg(void);
extern void _kdrCA3_reg(void);
extern void _kdr_BS_reg(void);
extern void _kdrinter_reg(void);
extern void _leak_reg(void);
extern void _nainter_reg(void);
extern void _napCA3_reg(void);
extern void _natCA3_reg(void);
extern void _nax_BS_reg(void);
extern void _vecevent_coreneuron_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"AMPA_NMDA_STP.mod\"");
    fprintf(stderr, " \"GABA_A_STP.mod\"");
    fprintf(stderr, " \"Gfluct.mod\"");
    fprintf(stderr, " \"cadad.mod\"");
    fprintf(stderr, " \"cal2.mod\"");
    fprintf(stderr, " \"can_mig.mod\"");
    fprintf(stderr, " \"exp2syn_stp.mod\"");
    fprintf(stderr, " \"gap.mod\"");
    fprintf(stderr, " \"h_kole.mod\"");
    fprintf(stderr, " \"imCA3.mod\"");
    fprintf(stderr, " \"kBK.mod\"");
    fprintf(stderr, " \"kap_BS.mod\"");
    fprintf(stderr, " \"kdmc_BS.mod\"");
    fprintf(stderr, " \"kdrCA3.mod\"");
    fprintf(stderr, " \"kdr_BS.mod\"");
    fprintf(stderr, " \"kdrinter.mod\"");
    fprintf(stderr, " \"leak.mod\"");
    fprintf(stderr, " \"nainter.mod\"");
    fprintf(stderr, " \"napCA3.mod\"");
    fprintf(stderr, " \"natCA3.mod\"");
    fprintf(stderr, " \"nax_BS.mod\"");
    fprintf(stderr, " \"vecevent_coreneuron.mod\"");
    fprintf(stderr, "\n");
  }
  _AMPA_NMDA_STP_reg();
  _GABA_A_STP_reg();
  _Gfluct_reg();
  _cadad_reg();
  _cal2_reg();
  _can_mig_reg();
  _exp2syn_stp_reg();
  _gap_reg();
  _h_kole_reg();
  _imCA3_reg();
  _kBK_reg();
  _kap_BS_reg();
  _kdmc_BS_reg();
  _kdrCA3_reg();
  _kdr_BS_reg();
  _kdrinter_reg();
  _leak_reg();
  _nainter_reg();
  _napCA3_reg();
  _natCA3_reg();
  _nax_BS_reg();
  _vecevent_coreneuron_reg();
}

#if defined(__cplusplus)
}
#endif
