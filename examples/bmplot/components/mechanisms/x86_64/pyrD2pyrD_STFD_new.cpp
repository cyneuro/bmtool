/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#define _pval pval
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if !NRNGPU
#undef exp
#define exp hoc_Exp
#endif
 
#define nrn_init _nrn_init__pyrD2pyrD_STFD
#define _nrn_initial _nrn_initial__pyrD2pyrD_STFD
#define nrn_cur _nrn_cur__pyrD2pyrD_STFD
#define _nrn_current _nrn_current__pyrD2pyrD_STFD
#define nrn_jacob _nrn_jacob__pyrD2pyrD_STFD
#define nrn_state _nrn_state__pyrD2pyrD_STFD
#define _net_receive _net_receive__pyrD2pyrD_STFD 
#define release release__pyrD2pyrD_STFD 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *hoc_getarg(int);
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define srcid _p[0]
#define srcid_columnindex 0
#define destid _p[1]
#define destid_columnindex 1
#define type _p[2]
#define type_columnindex 2
#define Cdur_nmda _p[3]
#define Cdur_nmda_columnindex 3
#define AlphaTmax_nmda _p[4]
#define AlphaTmax_nmda_columnindex 4
#define Beta_nmda _p[5]
#define Beta_nmda_columnindex 5
#define Erev_nmda _p[6]
#define Erev_nmda_columnindex 6
#define gbar_nmda _p[7]
#define gbar_nmda_columnindex 7
#define Cdur_ampa _p[8]
#define Cdur_ampa_columnindex 8
#define AlphaTmax_ampa _p[9]
#define AlphaTmax_ampa_columnindex 9
#define Beta_ampa _p[10]
#define Beta_ampa_columnindex 10
#define Erev_ampa _p[11]
#define Erev_ampa_columnindex 11
#define gbar_ampa _p[12]
#define gbar_ampa_columnindex 12
#define Cainf _p[13]
#define Cainf_columnindex 13
#define pooldiam _p[14]
#define pooldiam_columnindex 14
#define z _p[15]
#define z_columnindex 15
#define neuroM _p[16]
#define neuroM_columnindex 16
#define tauCa _p[17]
#define tauCa_columnindex 17
#define P0 _p[18]
#define P0_columnindex 18
#define fCa _p[19]
#define fCa_columnindex 19
#define lambda1 _p[20]
#define lambda1_columnindex 20
#define lambda2 _p[21]
#define lambda2_columnindex 21
#define threshold1 _p[22]
#define threshold1_columnindex 22
#define threshold2 _p[23]
#define threshold2_columnindex 23
#define initW _p[24]
#define initW_columnindex 24
#define fmax _p[25]
#define fmax_columnindex 25
#define fmin _p[26]
#define fmin_columnindex 26
#define thr_rp _p[27]
#define thr_rp_columnindex 27
#define facfactor _p[28]
#define facfactor_columnindex 28
#define f _p[29]
#define f_columnindex 29
#define tauF _p[30]
#define tauF_columnindex 30
#define d1 _p[31]
#define d1_columnindex 31
#define tauD1 _p[32]
#define tauD1_columnindex 32
#define d2 _p[33]
#define d2_columnindex 33
#define tauD2 _p[34]
#define tauD2_columnindex 34
#define ACH _p[35]
#define ACH_columnindex 35
#define bACH _p[36]
#define bACH_columnindex 36
#define inmda _p[37]
#define inmda_columnindex 37
#define g_nmda _p[38]
#define g_nmda_columnindex 38
#define on_nmda _p[39]
#define on_nmda_columnindex 39
#define W_nmda _p[40]
#define W_nmda_columnindex 40
#define iampa _p[41]
#define iampa_columnindex 41
#define g_ampa _p[42]
#define g_ampa_columnindex 42
#define on_ampa _p[43]
#define on_ampa_columnindex 43
#define limitW _p[44]
#define limitW_columnindex 44
#define ICa _p[45]
#define ICa_columnindex 45
#define iCatotal _p[46]
#define iCatotal_columnindex 46
#define Wmax _p[47]
#define Wmax_columnindex 47
#define Wmin _p[48]
#define Wmin_columnindex 48
#define maxChange _p[49]
#define maxChange_columnindex 49
#define normW _p[50]
#define normW_columnindex 50
#define scaleW _p[51]
#define scaleW_columnindex 51
#define tempW _p[52]
#define tempW_columnindex 52
#define pregid _p[53]
#define pregid_columnindex 53
#define postgid _p[54]
#define postgid_columnindex 54
#define F _p[55]
#define F_columnindex 55
#define D1 _p[56]
#define D1_columnindex 56
#define D2 _p[57]
#define D2_columnindex 57
#define r_nmda _p[58]
#define r_nmda_columnindex 58
#define r_ampa _p[59]
#define r_ampa_columnindex 59
#define capoolcon _p[60]
#define capoolcon_columnindex 60
#define W _p[61]
#define W_columnindex 61
#define eca _p[62]
#define eca_columnindex 62
#define t0 _p[63]
#define t0_columnindex 63
#define Afactor _p[64]
#define Afactor_columnindex 64
#define dW_ampa _p[65]
#define dW_ampa_columnindex 65
#define rp _p[66]
#define rp_columnindex 66
#define tsyn _p[67]
#define tsyn_columnindex 67
#define fa _p[68]
#define fa_columnindex 68
#define Dr_nmda _p[69]
#define Dr_nmda_columnindex 69
#define Dr_ampa _p[70]
#define Dr_ampa_columnindex 70
#define Dcapoolcon _p[71]
#define Dcapoolcon_columnindex 71
#define DW _p[72]
#define DW_columnindex 72
#define v _p[73]
#define v_columnindex 73
#define _g _p[74]
#define _g_columnindex 74
#define _tsav _p[75]
#define _tsav_columnindex 75
#define _nd_area  *_ppvar[0].get<double*>()
#define _ion_eca	*(_ppvar[2].get<double*>())
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_DA2(void*);
 static double _hoc_DA1(void*);
 static double _hoc_eta(void*);
 static double _hoc_omega(void*);
 static double _hoc_sfunc(void*);
 static double _hoc_unirand(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mechtype);
#endif
 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 {0, 0}
};
 static Member_func _member_func[] = {
 {"loc", _hoc_loc_pnt},
 {"has_loc", _hoc_has_loc},
 {"get_loc", _hoc_get_loc_pnt},
 {"DA2", _hoc_DA2},
 {"DA1", _hoc_DA1},
 {"eta", _hoc_eta},
 {"omega", _hoc_omega},
 {"sfunc", _hoc_sfunc},
 {"unirand", _hoc_unirand},
 {0, 0}
};
#define DA2 DA2_pyrD2pyrD_STFD
#define DA1 DA1_pyrD2pyrD_STFD
#define eta eta_pyrD2pyrD_STFD
#define omega omega_pyrD2pyrD_STFD
#define sfunc sfunc_pyrD2pyrD_STFD
#define unirand unirand_pyrD2pyrD_STFD
 extern double DA2( _threadargsprotocomma_ double , double );
 extern double DA1( _threadargsprotocomma_ double , double );
 extern double eta( _threadargsprotocomma_ double );
 extern double omega( _threadargsprotocomma_ double , double , double );
 extern double sfunc( _threadargsprotocomma_ double );
 extern double unirand( _threadargsproto_ );
 /* declare global and static user variables */
#define Beta2 Beta2_pyrD2pyrD_STFD
 double Beta2 = 0.0001;
#define Beta1 Beta1_pyrD2pyrD_STFD
 double Beta1 = 0.001;
#define DA_S DA_S_pyrD2pyrD_STFD
 double DA_S = 1.3;
#define DA_t3 DA_t3_pyrD2pyrD_STFD
 double DA_t3 = 0.9;
#define DA_t2 DA_t2_pyrD2pyrD_STFD
 double DA_t2 = 0.8;
#define DA_t1 DA_t1_pyrD2pyrD_STFD
 double DA_t1 = 1.2;
#define DAstop2 DAstop2_pyrD2pyrD_STFD
 double DAstop2 = 36000;
#define DAstart2 DAstart2_pyrD2pyrD_STFD
 double DAstart2 = 35900;
#define DAstop1 DAstop1_pyrD2pyrD_STFD
 double DAstop1 = 40000;
#define DAstart1 DAstart1_pyrD2pyrD_STFD
 double DAstart1 = 39500;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {"d2", 0, 1},
 {"d1", 0, 1},
 {"f", 0, 1e+09},
 {"tauD2", 1e-09, 1e+09},
 {"tauD1", 1e-09, 1e+09},
 {"tauF", 1e-09, 1e+09},
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"Beta1_pyrD2pyrD_STFD", "/ms"},
 {"Beta2_pyrD2pyrD_STFD", "/ms"},
 {"srcid", "1"},
 {"destid", "1"},
 {"Cdur_nmda", "ms"},
 {"AlphaTmax_nmda", "/ms"},
 {"Beta_nmda", "/ms"},
 {"Erev_nmda", "mV"},
 {"gbar_nmda", "uS"},
 {"Cdur_ampa", "ms"},
 {"AlphaTmax_ampa", "/ms"},
 {"Beta_ampa", "/ms"},
 {"Erev_ampa", "mV"},
 {"gbar_ampa", "uS"},
 {"Cainf", "mM"},
 {"pooldiam", "micrometer"},
 {"tauCa", "ms"},
 {"f", "1"},
 {"tauF", "ms"},
 {"d1", "1"},
 {"tauD1", "ms"},
 {"d2", "1"},
 {"tauD2", "ms"},
 {"inmda", "nA"},
 {"g_nmda", "uS"},
 {"iampa", "nA"},
 {"g_ampa", "uS"},
 {"ICa", "mA"},
 {"iCatotal", "mA"},
 {0, 0}
};
 static double W0 = 0;
 static double capoolcon0 = 0;
 static double delta_t = 0.01;
 static double r_ampa0 = 0;
 static double r_nmda0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"DAstart1_pyrD2pyrD_STFD", &DAstart1_pyrD2pyrD_STFD},
 {"DAstop1_pyrD2pyrD_STFD", &DAstop1_pyrD2pyrD_STFD},
 {"DAstart2_pyrD2pyrD_STFD", &DAstart2_pyrD2pyrD_STFD},
 {"DAstop2_pyrD2pyrD_STFD", &DAstop2_pyrD2pyrD_STFD},
 {"DA_t1_pyrD2pyrD_STFD", &DA_t1_pyrD2pyrD_STFD},
 {"DA_t2_pyrD2pyrD_STFD", &DA_t2_pyrD2pyrD_STFD},
 {"DA_t3_pyrD2pyrD_STFD", &DA_t3_pyrD2pyrD_STFD},
 {"DA_S_pyrD2pyrD_STFD", &DA_S_pyrD2pyrD_STFD},
 {"Beta1_pyrD2pyrD_STFD", &Beta1_pyrD2pyrD_STFD},
 {"Beta2_pyrD2pyrD_STFD", &Beta2_pyrD2pyrD_STFD},
 {0, 0}
};
 static DoubVec hoc_vdoub[] = {
 {0, 0, 0}
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, Memb_list*, int);
static void nrn_state(NrnThread*, Memb_list*, int);
 static void nrn_cur(NrnThread*, Memb_list*, int);
static void  nrn_jacob(NrnThread*, Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, Memb_list*, int);
static void _ode_matsol(NrnThread*, Memb_list*, int);
 
#define _cvode_ieq _ppvar[4].literal_value<int>()
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"pyrD2pyrD_STFD",
 "srcid",
 "destid",
 "type",
 "Cdur_nmda",
 "AlphaTmax_nmda",
 "Beta_nmda",
 "Erev_nmda",
 "gbar_nmda",
 "Cdur_ampa",
 "AlphaTmax_ampa",
 "Beta_ampa",
 "Erev_ampa",
 "gbar_ampa",
 "Cainf",
 "pooldiam",
 "z",
 "neuroM",
 "tauCa",
 "P0",
 "fCa",
 "lambda1",
 "lambda2",
 "threshold1",
 "threshold2",
 "initW",
 "fmax",
 "fmin",
 "thr_rp",
 "facfactor",
 "f",
 "tauF",
 "d1",
 "tauD1",
 "d2",
 "tauD2",
 "ACH",
 "bACH",
 0,
 "inmda",
 "g_nmda",
 "on_nmda",
 "W_nmda",
 "iampa",
 "g_ampa",
 "on_ampa",
 "limitW",
 "ICa",
 "iCatotal",
 "Wmax",
 "Wmin",
 "maxChange",
 "normW",
 "scaleW",
 "tempW",
 "pregid",
 "postgid",
 "F",
 "D1",
 "D2",
 0,
 "r_nmda",
 "r_ampa",
 "capoolcon",
 "W",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 76, _prop);
 	/*initialize range parameters*/
 	srcid = -1;
 	destid = -1;
 	type = -1;
 	Cdur_nmda = 16.765;
 	AlphaTmax_nmda = 0.2659;
 	Beta_nmda = 0.008;
 	Erev_nmda = 0;
 	gbar_nmda = 0.0005;
 	Cdur_ampa = 1.421;
 	AlphaTmax_ampa = 3.8142;
 	Beta_ampa = 0.1429;
 	Erev_ampa = 0;
 	gbar_ampa = 0.001;
 	Cainf = 5e-05;
 	pooldiam = 1.8172;
 	z = 2;
 	neuroM = 0;
 	tauCa = 50;
 	P0 = 0.015;
 	fCa = 0.024;
 	lambda1 = 40;
 	lambda2 = 0.03;
 	threshold1 = 0.4;
 	threshold2 = 0.55;
 	initW = 5;
 	fmax = 3;
 	fmin = 0.8;
 	thr_rp = 1;
 	facfactor = 1;
 	f = 0;
 	tauF = 20;
 	d1 = 0.95;
 	tauD1 = 40;
 	d2 = 0.9;
 	tauD2 = 70;
 	ACH = 1;
 	bACH = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 76;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[2] = &prop_ion->param[0]; /* eca */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 
#define _tqitem &(_ppvar[3])
 static void _net_receive(Point_process*, double*, double);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _pyrD2pyrD_STFD_new_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
  hoc_register_prop_size(_mechtype, 76, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "netsend");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 pyrD2pyrD_STFD /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/pyrD2pyrD_STFD_new.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96485.0;
 static double pilocal = 3.141592;
static int _reset;
static const char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[4], _dlist1[4];
 static int release(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DW = 1e-12 * limitW * eta ( _threadargscomma_ capoolcon ) * ( lambda1 * omega ( _threadargscomma_ capoolcon , threshold1 , threshold2 ) - lambda2 * W ) ;
   Dr_nmda = AlphaTmax_nmda * on_nmda * ( 1.0 - r_nmda ) - Beta_nmda * r_nmda ;
   Dr_ampa = AlphaTmax_ampa * on_ampa * ( 1.0 - r_ampa ) - Beta_ampa * r_ampa ;
   Dcapoolcon = - fCa * Afactor * ICa + ( Cainf - capoolcon ) / tauCa ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DW = DW  / (1. - dt*( ( 1e-12 * limitW * eta ( _threadargscomma_ capoolcon ) )*( ( ( - ( lambda2 )*( 1.0 ) ) ) ) )) ;
 Dr_nmda = Dr_nmda  / (1. - dt*( ( AlphaTmax_nmda * on_nmda )*( ( ( - 1.0 ) ) ) - ( Beta_nmda )*( 1.0 ) )) ;
 Dr_ampa = Dr_ampa  / (1. - dt*( ( AlphaTmax_ampa * on_ampa )*( ( ( - 1.0 ) ) ) - ( Beta_ampa )*( 1.0 ) )) ;
 Dcapoolcon = Dcapoolcon  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tauCa )) ;
  return 0;
}
 /*END CVODE*/
 static int release (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    W = W + (1. - exp(dt*(( 1e-12 * limitW * eta ( _threadargscomma_ capoolcon ) )*( ( ( - ( lambda2 )*( 1.0 ) ) ) ))))*(- ( ( ( ( 1e-12 )*( limitW ) )*( eta ( _threadargscomma_ capoolcon ) ) )*( ( ( lambda1 )*( omega ( _threadargscomma_ capoolcon , threshold1 , threshold2 ) ) ) ) ) / ( ( ( ( 1e-12 )*( limitW ) )*( eta ( _threadargscomma_ capoolcon ) ) )*( ( ( - ( lambda2 )*( 1.0 ) ) ) ) ) - W) ;
    r_nmda = r_nmda + (1. - exp(dt*(( AlphaTmax_nmda * on_nmda )*( ( ( - 1.0 ) ) ) - ( Beta_nmda )*( 1.0 ))))*(- ( ( ( AlphaTmax_nmda )*( on_nmda ) )*( ( 1.0 ) ) ) / ( ( ( AlphaTmax_nmda )*( on_nmda ) )*( ( ( - 1.0 ) ) ) - ( Beta_nmda )*( 1.0 ) ) - r_nmda) ;
    r_ampa = r_ampa + (1. - exp(dt*(( AlphaTmax_ampa * on_ampa )*( ( ( - 1.0 ) ) ) - ( Beta_ampa )*( 1.0 ))))*(- ( ( ( AlphaTmax_ampa )*( on_ampa ) )*( ( 1.0 ) ) ) / ( ( ( AlphaTmax_ampa )*( on_ampa ) )*( ( ( - 1.0 ) ) ) - ( Beta_ampa )*( 1.0 ) ) - r_ampa) ;
    capoolcon = capoolcon + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tauCa)))*(- ( ( ( - fCa )*( Afactor ) )*( ICa ) + ( ( Cainf ) ) / tauCa ) / ( ( ( ( - 1.0 ) ) ) / tauCa ) - capoolcon) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = nullptr;}
 {
   if ( _lflag  == 0.0 ) {
     if ( (  ! on_nmda ) ) {
       t0 = t ;
       on_nmda = 1.0 ;
       net_send ( _tqitem, _args, _pnt, t +  Cdur_nmda , 1.0 ) ;
       }
     else if ( on_nmda  == 1.0 ) {
       net_move ( _tqitem, _pnt, t + Cdur_nmda ) ;
       t0 = t ;
       }
     }
   if ( _lflag  == 1.0 ) {
     on_nmda = 0.0 ;
     }
   if ( _lflag  == 0.0 ) {
     rp = unirand ( _threadargs_ ) ;
     D1 = 1.0 - ( 1.0 - D1 ) * exp ( - ( t - tsyn ) / tauD1 ) ;
     D2 = 1.0 - ( 1.0 - D2 ) * exp ( - ( t - tsyn ) / tauD2 ) ;
     tsyn = t ;
     facfactor = F * D1 * D2 ;
     if ( F > 3.0 ) {
       F = 3.0 ;
       }
     if ( facfactor < 0.5 ) {
       facfactor = 0.5 ;
       }
     D1 = D1 * d1 ;
     D2 = D2 * d2 ;
     }
   } }
 
double sfunc ( _threadargsprotocomma_ double _lv ) {
   double _lsfunc;
  _lsfunc = 1.0 / ( 1.0 + 0.33 * exp ( - 0.06 * _lv ) ) ;
    
return _lsfunc;
 }
 
static double _hoc_sfunc(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  sfunc ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
double eta ( _threadargsprotocomma_ double _lCani ) {
   double _leta;
 double _ltaulearn , _lP1 , _lP2 , _lP4 , _lCacon ;
 _lP1 = 0.1 ;
   _lP2 = _lP1 * 1e-4 ;
   _lP4 = 1.0 ;
   _lCacon = _lCani * 1e3 ;
   _ltaulearn = _lP1 / ( _lP2 + _lCacon * _lCacon * _lCacon ) + _lP4 ;
   _leta = 1.0 / _ltaulearn * 0.001 ;
   
return _leta;
 }
 
static double _hoc_eta(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  eta ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
double omega ( _threadargsprotocomma_ double _lCani , double _lthreshold1 , double _lthreshold2 ) {
   double _lomega;
 double _lr , _lmid , _lCacon ;
 _lCacon = _lCani * 1e3 ;
   _lr = ( _lthreshold2 - _lthreshold1 ) / 2.0 ;
   _lmid = ( _lthreshold1 + _lthreshold2 ) / 2.0 ;
   if ( _lCacon <= _lthreshold1 ) {
     _lomega = 0.0 ;
     }
   else if ( _lCacon >= _lthreshold2 ) {
     _lomega = 1.0 / ( 1.0 + 50.0 * exp ( - 50.0 * ( _lCacon - _lthreshold2 ) ) ) ;
     }
   else {
     _lomega = - sqrt ( _lr * _lr - ( _lCacon - _lmid ) * ( _lCacon - _lmid ) ) ;
     }
   
return _lomega;
 }
 
static double _hoc_omega(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  omega ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) );
 return(_r);
}
 
double DA1 ( _threadargsprotocomma_ double _lDAstart1 , double _lDAstop1 ) {
   double _lDA1;
 double _lDAtemp1 , _lDAtemp2 , _lDAtemp3 , _lDAtemp4 , _lDAtemp5 , _lDAtemp6 , _lDAtemp7 , _lDAtemp8 , _lDAtemp9 , _lDAtemp10 , _lDAtemp11 , _lDAtemp12 , _lDAtemp13 , _lDAtemp14 , _lDAtemp15 , _lDAtemp16 , _lDAtemp17 , _lDAtemp18 , _lDAtemp19 , _lDAtemp20 , _lDAtemp21 , _lDAtemp22 , _lDAtemp23 , _lDAtemp24 , _lDAtemp25 , _lDAtemp26 , _lDAtemp27 , _lDAtemp28 , _lDAtemp29 , _lDAtemp30 , _lDAtemp31 , _lDAtemp32 , _lDAtemp33 , _lDAtemp34 , _ls ;
 _lDAtemp1 = _lDAstart1 + 4000.0 ;
   _lDAtemp2 = _lDAtemp1 + 4000.0 ;
   _lDAtemp3 = _lDAtemp2 + 4000.0 ;
   _lDAtemp4 = _lDAtemp3 + 4000.0 ;
   _lDAtemp5 = _lDAtemp4 + 4000.0 ;
   _lDAtemp6 = _lDAtemp5 + 4000.0 ;
   _lDAtemp7 = _lDAtemp6 + 4000.0 ;
   _lDAtemp8 = _lDAtemp7 + 4000.0 ;
   _lDAtemp9 = _lDAtemp8 + 4000.0 ;
   _lDAtemp10 = _lDAtemp9 + 4000.0 ;
   _lDAtemp11 = _lDAtemp10 + 4000.0 ;
   _lDAtemp12 = _lDAtemp11 + 4000.0 ;
   _lDAtemp13 = _lDAtemp12 + 4000.0 ;
   _lDAtemp14 = _lDAtemp13 + 4000.0 ;
   _lDAtemp15 = _lDAtemp14 + 4000.0 + 100000.0 ;
   _lDAtemp16 = _lDAtemp15 + 4000.0 ;
   _lDAtemp17 = _lDAtemp16 + 4000.0 ;
   _lDAtemp18 = _lDAtemp17 + 4000.0 ;
   _lDAtemp19 = _lDAtemp18 + 4000.0 ;
   _lDAtemp20 = _lDAtemp19 + 4000.0 ;
   _lDAtemp21 = _lDAtemp20 + 4000.0 ;
   _lDAtemp22 = _lDAtemp21 + 4000.0 ;
   _lDAtemp23 = _lDAtemp22 + 4000.0 ;
   _lDAtemp24 = _lDAtemp23 + 4000.0 ;
   _lDAtemp25 = _lDAtemp24 + 4000.0 ;
   _lDAtemp26 = _lDAtemp25 + 4000.0 ;
   _lDAtemp27 = _lDAtemp26 + 4000.0 ;
   _lDAtemp28 = _lDAtemp27 + 4000.0 ;
   _lDAtemp29 = _lDAtemp28 + 4000.0 ;
   _lDAtemp30 = _lDAtemp29 + 4000.0 ;
   _lDAtemp31 = _lDAtemp30 + 4000.0 ;
   _lDAtemp32 = _lDAtemp31 + 4000.0 ;
   _lDAtemp33 = _lDAtemp32 + 4000.0 ;
   _lDAtemp34 = _lDAtemp33 + 4000.0 ;
   if ( t <= _lDAstart1 ) {
     _lDA1 = 1.0 ;
     }
   else if ( t >= _lDAstart1  && t <= _lDAstop1 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAstop1  && t < _lDAtemp1 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - _lDAstop1 ) ) ;
     }
   else if ( t >= _lDAtemp1  && t <= _lDAtemp1 + 500.0 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAtemp1 + 500.0  && t < _lDAtemp2 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lDAtemp1 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp2  && t <= _lDAtemp2 + 500.0 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAtemp2 + 500.0  && t < _lDAtemp3 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lDAtemp2 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp3  && t <= _lDAtemp3 + 500.0 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAtemp3 + 500.0  && t < _lDAtemp4 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lDAtemp3 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp4  && t <= _lDAtemp4 + 500.0 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAtemp4 + 500.0  && t < _lDAtemp5 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lDAtemp4 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp5  && t <= _lDAtemp5 + 500.0 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAtemp5 + 500.0  && t < _lDAtemp6 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lDAtemp5 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp6  && t <= _lDAtemp6 + 500.0 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAtemp6 + 500.0  && t < _lDAtemp7 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lDAtemp6 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp7  && t <= _lDAtemp7 + 500.0 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAtemp7 + 500.0  && t < _lDAtemp8 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lDAtemp7 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp8  && t <= _lDAtemp8 + 500.0 ) {
     _lDA1 = DA_t1 ;
     }
   else if ( t > _lDAtemp8 + 500.0  && t < _lDAtemp9 ) {
     _lDA1 = 1.0 + ( DA_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lDAtemp8 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp9  && t <= _lDAtemp9 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp9 + 500.0  && t < _lDAtemp10 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp9 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp10  && t <= _lDAtemp10 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp10 + 500.0  && t < _lDAtemp11 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp10 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp11  && t <= _lDAtemp11 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp11 + 500.0  && t < _lDAtemp12 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp11 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp12  && t <= _lDAtemp12 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp12 + 500.0  && t < _lDAtemp13 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp12 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp13  && t <= _lDAtemp13 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp13 + 500.0  && t < _lDAtemp14 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp13 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp14  && t <= _lDAtemp14 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp14 + 500.0  && t < _lDAtemp15 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp14 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp15  && t <= _lDAtemp15 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp15 + 500.0  && t < _lDAtemp16 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp15 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp16  && t <= _lDAtemp16 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp16 + 500.0  && t < _lDAtemp17 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp16 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp17  && t <= _lDAtemp17 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp17 + 500.0  && t < _lDAtemp18 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp17 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp18  && t <= _lDAtemp18 + 500.0 ) {
     _lDA1 = DA_t2 ;
     }
   else if ( t > _lDAtemp18 + 500.0  && t < _lDAtemp19 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp18 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp19  && t <= _lDAtemp19 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp19 + 500.0  && t < _lDAtemp20 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp19 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp20  && t <= _lDAtemp20 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp20 + 500.0  && t < _lDAtemp21 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp20 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp21  && t <= _lDAtemp21 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp21 + 500.0  && t < _lDAtemp22 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp21 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp22  && t <= _lDAtemp22 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp22 + 500.0  && t < _lDAtemp23 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp22 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp23  && t <= _lDAtemp23 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp23 + 500.0  && t < _lDAtemp24 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp23 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp24  && t <= _lDAtemp24 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp24 + 500.0  && t < _lDAtemp25 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp24 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp25  && t <= _lDAtemp25 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp25 + 500.0  && t < _lDAtemp26 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp25 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp26  && t <= _lDAtemp26 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp26 + 500.0  && t < _lDAtemp27 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp26 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp27  && t <= _lDAtemp27 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp27 + 500.0  && t < _lDAtemp28 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp27 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp28  && t <= _lDAtemp28 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp28 + 500.0  && t < _lDAtemp29 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp28 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp29  && t <= _lDAtemp29 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp29 + 500.0  && t < _lDAtemp30 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp29 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp30  && t <= _lDAtemp30 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp30 + 500.0  && t < _lDAtemp31 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp30 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp31  && t <= _lDAtemp31 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp31 + 500.0  && t < _lDAtemp32 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp31 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp32  && t <= _lDAtemp32 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp32 + 500.0  && t < _lDAtemp33 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp32 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp33  && t <= _lDAtemp33 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else if ( t > _lDAtemp33 + 500.0  && t < _lDAtemp34 ) {
     _lDA1 = 1.0 + ( DA_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAtemp33 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDAtemp34  && t <= _lDAtemp34 + 500.0 ) {
     _lDA1 = DA_t3 ;
     }
   else {
     _lDA1 = 1.0 ;
     }
   
return _lDA1;
 }
 
static double _hoc_DA1(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  DA1 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}
 
double DA2 ( _threadargsprotocomma_ double _lDAstart2 , double _lDAstop2 ) {
   double _lDA2;
 double _lDA2temp1 , _lDA2temp2 , _lDA2temp3 , _lDA2temp4 , _lDA2temp5 , _lDA2temp6 , _lDA2temp7 , _lDA2temp8 , _lDA2temp9 , _lDA2temp10 , _lDA2temp11 , _lDA2temp12 , _lDA2temp13 , _lDA2temp14 , _lDA2temp15 , _lDA2temp16 , _ls ;
 _lDA2temp1 = _lDAstart2 + 4000.0 ;
   _lDA2temp2 = _lDA2temp1 + 4000.0 ;
   _lDA2temp3 = _lDA2temp2 + 4000.0 ;
   _lDA2temp4 = _lDA2temp3 + 4000.0 ;
   _lDA2temp5 = _lDA2temp4 + 4000.0 ;
   _lDA2temp6 = _lDA2temp5 + 4000.0 ;
   _lDA2temp7 = _lDA2temp6 + 4000.0 ;
   _lDA2temp8 = _lDA2temp7 + 4000.0 ;
   _lDA2temp9 = _lDA2temp8 + 4000.0 ;
   _lDA2temp10 = _lDA2temp9 + 4000.0 ;
   _lDA2temp11 = _lDA2temp10 + 4000.0 ;
   _lDA2temp12 = _lDA2temp11 + 4000.0 ;
   _lDA2temp13 = _lDA2temp12 + 4000.0 ;
   _lDA2temp14 = _lDA2temp13 + 4000.0 ;
   _lDA2temp15 = _lDA2temp14 + 4000.0 ;
   if ( t <= _lDAstart2 ) {
     _lDA2 = 1.0 ;
     }
   else if ( t >= _lDAstart2  && t <= _lDAstop2 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDAstop2  && t < _lDA2temp1 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDAstop2 + 500.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp1  && t <= _lDA2temp1 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp1 + 100.0  && t < _lDA2temp2 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp1 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp2  && t <= _lDA2temp2 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp2 + 100.0  && t < _lDA2temp3 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp2 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp3  && t <= _lDA2temp3 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp3 + 100.0  && t < _lDA2temp4 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp3 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp4  && t <= _lDA2temp4 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp4 + 100.0  && t < _lDA2temp5 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp4 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp5  && t <= _lDA2temp5 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp5 + 100.0  && t < _lDA2temp6 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp5 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp6  && t <= _lDA2temp6 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp6 + 100.0  && t < _lDA2temp7 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp6 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp7  && t <= _lDA2temp7 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp7 + 100.0  && t < _lDA2temp8 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp7 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp8  && t <= _lDA2temp8 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp8 + 100.0  && t < _lDA2temp9 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp8 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp9  && t <= _lDA2temp9 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp9 + 100.0  && t < _lDA2temp10 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp9 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp10  && t <= _lDA2temp10 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp10 + 100.0  && t < _lDA2temp11 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp10 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp11  && t <= _lDA2temp11 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp11 + 100.0  && t < _lDA2temp12 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp11 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp12  && t <= _lDA2temp12 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp12 + 100.0  && t < _lDA2temp13 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp12 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp13  && t <= _lDA2temp13 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp13 + 100.0  && t < _lDA2temp14 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp13 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp14  && t <= _lDA2temp14 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else if ( t > _lDA2temp14 + 100.0  && t < _lDA2temp15 ) {
     _lDA2 = 1.0 + ( DA_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lDA2temp14 + 100.0 ) ) ) ;
     }
   else if ( t >= _lDA2temp15  && t <= _lDA2temp15 + 100.0 ) {
     _lDA2 = DA_S ;
     }
   else {
     _lDA2 = 1.0 ;
     }
   
return _lDA2;
 }
 
static double _hoc_DA2(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  DA2 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}
 
double unirand ( _threadargsproto_ ) {
   double _lunirand;
 _lunirand = 0.0 ;
   
return _lunirand;
 }
 
static double _hoc_unirand(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  unirand ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int _ode_count(int _type){ return 4;}
 
static void _ode_spec(NrnThread* _nt, Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  eca = _ion_eca;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 4; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  eca = _ion_eca;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 0);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  W = W0;
  capoolcon = capoolcon0;
  r_ampa = r_ampa0;
  r_nmda = r_nmda0;
 {
   on_nmda = 0.0 ;
   r_nmda = 0.0 ;
   W_nmda = initW ;
   on_ampa = 0.0 ;
   r_ampa = 0.0 ;
   W = initW ;
   limitW = 1.0 ;
   tempW = initW ;
   t0 = - 1.0 ;
   Wmax = fmax * initW ;
   Wmin = fmin * initW ;
   maxChange = ( Wmax - Wmin ) / 10.0 ;
   dW_ampa = 0.0 ;
   capoolcon = Cainf ;
   Afactor = 1.0 / ( z * FARADAY * 4.0 / 3.0 * pilocal * pow( ( pooldiam / 2.0 ) , 3.0 ) ) * ( 1e6 ) ;
   fa = 0.0 ;
   F = 1.0 ;
   D1 = 1.0 ;
   D2 = 1.0 ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  eca = _ion_eca;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   if ( ( eta ( _threadargscomma_ capoolcon ) * ( lambda1 * omega ( _threadargscomma_ capoolcon , threshold1 , threshold2 ) - lambda2 * W ) ) > 0.0  && W >= Wmax ) {
     limitW = 1e-12 ;
     }
   else if ( ( eta ( _threadargscomma_ capoolcon ) * ( lambda1 * omega ( _threadargscomma_ capoolcon , threshold1 , threshold2 ) - lambda2 * W ) ) < 0.0  && W <= Wmin ) {
     limitW = 1e-12 ;
     }
   else {
     limitW = 1.0 ;
     }
   if ( t0 > 0.0 ) {
     if ( rp < thr_rp ) {
       if ( t - t0 < Cdur_ampa ) {
         on_ampa = 1.0 ;
         }
       else {
         on_ampa = 0.0 ;
         }
       }
     else {
       on_ampa = 0.0 ;
       }
     }
   if ( neuroM  == 1.0 ) {
     g_nmda = gbar_nmda * r_nmda * facfactor * DA1 ( _threadargscomma_ DAstart1 , DAstop1 ) * DA2 ( _threadargscomma_ DAstart2 , DAstop2 ) ;
     }
   else {
     g_nmda = gbar_nmda * r_nmda * facfactor ;
     }
   inmda = W_nmda * g_nmda * ( v - Erev_nmda ) * sfunc ( _threadargscomma_ v ) ;
   g_ampa = gbar_ampa * r_ampa * facfactor ;
   iampa = W * g_ampa * ( v - Erev_ampa ) * ( 1.0 + ( bACH * ( ACH - 1.0 ) ) ) ;
   ICa = P0 * g_nmda * ( v - eca ) * sfunc ( _threadargscomma_ v ) ;
   }
 _current += inmda;
 _current += iampa;

} return _current;
}

static void nrn_cur(NrnThread* _nt, Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  eca = _ion_eca;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  eca = _ion_eca;
 {   release(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = W_columnindex;  _dlist1[0] = DW_columnindex;
 _slist1[1] = r_nmda_columnindex;  _dlist1[1] = Dr_nmda_columnindex;
 _slist1[2] = r_ampa_columnindex;  _dlist1[2] = Dr_ampa_columnindex;
 _slist1[3] = capoolcon_columnindex;  _dlist1[3] = Dcapoolcon_columnindex;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/pyrD2pyrD_STFD_new.mod";
    const char* nmodl_file_text = 
  ":Pyramidal Cells to Pyramidal Cells AMPA+NMDA with local Ca2+ pool\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS pyrD2pyrD_STFD\n"
  "	USEION ca READ eca	\n"
  "	NONSPECIFIC_CURRENT inmda, iampa\n"
  "	RANGE initW\n"
  "	RANGE Cdur_nmda, AlphaTmax_nmda, Beta_nmda, Erev_nmda, gbar_nmda, W_nmda, on_nmda, g_nmda\n"
  "	RANGE Cdur_ampa, AlphaTmax_ampa, Beta_ampa, Erev_ampa, gbar_ampa, W, on_ampa, g_ampa\n"
  "	RANGE eca, ICa, P0, fCa, tauCa, iCatotal\n"
  "	RANGE Cainf, pooldiam, z\n"
  "	RANGE lambda1, lambda2, threshold1, threshold2\n"
  "	RANGE fmax, fmin, Wmax, Wmin, maxChange, normW, scaleW, limitW, srcid,destid,tempW \n"
  "	RANGE pregid,postgid, thr_rp\n"
  "	RANGE F, f, tauF, D1, d1, tauD1, D2, d2, tauD2\n"
  "	RANGE facfactor\n"
  "	RANGE neuroM,type\n"
  "	RANGE bACH, ACH\n"
  "}\n"
  "\n"
  "UNITS { \n"
  "	(mV) = (millivolt)\n"
  "        (nA) = (nanoamp)\n"
  "	(uS) = (microsiemens)\n"
  "	FARADAY = 96485 (coul)\n"
  "	pilocal = 3.141592 (1)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	srcid = -1 (1)\n"
  "	destid = -1 (1)\n"
  "	type = -1\n"
  "	\n"
  "	Cdur_nmda = 16.7650 (ms)\n"
  "	AlphaTmax_nmda = .2659 (/ms)\n"
  "	Beta_nmda = 0.008 (/ms)\n"
  "	Erev_nmda = 0 (mV)\n"
  "	gbar_nmda = .5e-3 (uS)\n"
  "\n"
  "	Cdur_ampa = 1.4210 (ms)\n"
  "	AlphaTmax_ampa = 3.8142 (/ms)\n"
  "	Beta_ampa =  0.1429(/ms) :0.1429 as original 0.2858 as half,0.07145 as twice\n"
  "	Erev_ampa = 0 (mV)\n"
  "	gbar_ampa = 1e-3 (uS)\n"
  "\n"
  "	eca = 120\n"
  "\n"
  "	Cainf = 50e-6 (mM)\n"
  "	pooldiam =  1.8172 (micrometer)\n"
  "	z = 2\n"
  "	neuroM = 0\n"
  "	tauCa = 50 (ms)\n"
  "	P0 = .015\n"
  "	fCa = .024\n"
  "	\n"
  "	lambda1 = 40 : 60 : 12 :80: 20 : 15 :8 :5: 2.5\n"
  "	lambda2 = .03\n"
  "	threshold1 = 0.4 :  0.45 : 0.35 :0.35:0.2 :0.50 (uM)\n"
  "	threshold2 = 0.55 : 0.50 : 0.40 :0.4 :0.3 :0.60 (uM)\n"
  "\n"
  "	initW = 5.0 : 1.0 :  0.9 : 0.8 : 2 : 10 : 6 :1.5\n"
  "	fmax = 3 : 2.5 : 4 : 2 : 3 : 1.5 : 3\n"
  "	fmin = .8\n"
  "	\n"
  "	DAstart1 = 39500\n"
  "	DAstop1 = 40000	\n"
  "	DAstart2 = 35900\n"
  "	DAstop2 = 36000	\n"
  "\n"
  "	DA_t1 = 1.2\n"
  "	DA_t2 = 0.8 : 0.9\n"
  "    DA_t3 = 0.9\n"
  "	DA_S = 1.3 : 0.95 : 0.6	\n"
  "	Beta1 = 0.001  (/ms) : 1/decay time for neuromodulators\n"
  "	Beta2 = 0.0001  (/ms)\n"
  "\n"
  "	thr_rp = 1 : .7\n"
  "	\n"
  "	facfactor = 1\n"
  "	: the (1) is needed for the range limits to be effective\n"
  "        f = 0 (1) < 0, 1e9 >    : facilitation\n"
  "        tauF = 20 (ms) < 1e-9, 1e9 >\n"
  "        d1 = 0.95 (1) < 0, 1 >     : fast depression\n"
  "        tauD1 = 40 (ms) < 1e-9, 1e9 >\n"
  "        d2 = 0.9 (1) < 0, 1 >     : slow depression\n"
  "        tauD2 = 70 (ms) < 1e-9, 1e9 >		\n"
  "\n"
  "	ACH = 1\n"
  "	bACH = 0\n"
  "\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)\n"
  "\n"
  "	inmda (nA)\n"
  "	g_nmda (uS)\n"
  "	on_nmda\n"
  "	W_nmda\n"
  "\n"
  "	iampa (nA)\n"
  "	g_ampa (uS)\n"
  "	on_ampa\n"
  "	: W\n"
  "	limitW\n"
  "\n"
  "	t0 (ms)\n"
  "\n"
  "	ICa (mA)\n"
  "	Afactor	(mM/ms/nA)\n"
  "	iCatotal (mA)\n"
  "\n"
  "	dW_ampa\n"
  "	Wmax\n"
  "	Wmin\n"
  "	maxChange\n"
  "	normW\n"
  "	scaleW\n"
  "	\n"
  "    tempW\n"
  "	pregid\n"
  "	postgid\n"
  "\n"
  "	rp\n"
  "	tsyn\n"
  "	\n"
  "	fa\n"
  "	F\n"
  "	D1\n"
  "	D2\n"
  "}\n"
  "\n"
  "STATE { r_nmda r_ampa capoolcon W}\n"
  "\n"
  "INITIAL {\n"
  "	on_nmda = 0\n"
  "	r_nmda = 0\n"
  "	W_nmda = initW\n"
  "\n"
  "	on_ampa = 0\n"
  "	r_ampa = 0\n"
  "	W = initW\n"
  "    limitW = 1\n"
  "    \n"
  "	tempW = initW\n"
  "	t0 = -1\n"
  "\n"
  "	Wmax = fmax*initW\n"
  "	Wmin = fmin*initW\n"
  "	maxChange = (Wmax-Wmin)/10\n"
  "	dW_ampa = 0\n"
  "\n"
  "	capoolcon = Cainf\n"
  "	Afactor	= 1/(z*FARADAY*4/3*pilocal*(pooldiam/2)^3)*(1e6)\n"
  "\n"
  "	fa =0\n"
  "	F = 1\n"
  "	D1 = 1\n"
  "	D2 = 1\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "\n"
  "if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W))>0&&W>=Wmax) {\n"
  "        limitW=1e-12\n"
  "	} else if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W))<0&&W<=Wmin) {\n"
  "        limitW=1e-12\n"
  "	} else {\n"
  "	limitW=1 }\n"
  "	\n"
  "	SOLVE release METHOD cnexp\n"
  "	if (t0>0) {\n"
  "		if (rp < thr_rp) {\n"
  "			if (t-t0 < Cdur_ampa) {\n"
  "				on_ampa = 1\n"
  "			} else {\n"
  "				on_ampa = 0\n"
  "			}\n"
  "		} else {\n"
  "			on_ampa = 0\n"
  "		}\n"
  "	}\n"
  "          : if (W >= Wmax || W <= Wmin ) {     : for limiting the weight\n"
  "	 : limitW=1e-12\n"
  "	 : } else {\n"
  "	  : limitW=1\n"
  "	 : }\n"
  "	 \n"
  "	 :if (W > Wmax) { \n"
  "		:W = Wmax\n"
  "	:} else if (W < Wmin) {\n"
  " 		:W = Wmin\n"
  "	:}\n"
  "	 \n"
  "	if (neuroM==1) {\n"
  "	g_nmda = gbar_nmda*r_nmda*facfactor*DA1(DAstart1,DAstop1)*DA2(DAstart2,DAstop2)        : Dopamine effect on NMDA to reduce NMDA current amplitude\n"
  "		} else {\n"
  "		g_nmda = gbar_nmda*r_nmda*facfactor\n"
  "		}\n"
  "		inmda = W_nmda*g_nmda*(v - Erev_nmda)*sfunc(v)\n"
  "\n"
  "	g_ampa = gbar_ampa*r_ampa*facfactor\n"
  "	iampa = W*g_ampa*(v - Erev_ampa)*(1 + (bACH * (ACH - 1)))\n"
  "\n"
  "	ICa = P0*g_nmda*(v - eca)*sfunc(v)\n"
  "	\n"
  "}\n"
  "\n"
  "DERIVATIVE release {\n"
  "	: W' = eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)\n"
  "	\n"
  "	W' = 1e-12*limitW*eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)\n"
  "	r_nmda' = AlphaTmax_nmda*on_nmda*(1-r_nmda)-Beta_nmda*r_nmda\n"
  "	r_ampa' = AlphaTmax_ampa*on_ampa*(1-r_ampa)-Beta_ampa*r_ampa\n"
  "  	capoolcon'= -fCa*Afactor*ICa + (Cainf-capoolcon)/tauCa\n"
  "}\n"
  "\n"
  "NET_RECEIVE(dummy_weight) {\n"
  "	    if (flag==0) {           :a spike arrived, start onset state if not already on\n"
  "         if ((!on_nmda)){       :this synpase joins the set of synapses in onset state\n"
  "           t0=t\n"
  "	      on_nmda=1		\n"
  "	      net_send(Cdur_nmda,1)  \n"
  "         } else if (on_nmda==1) {             :already in onset state, so move offset time\n"
  "          net_move(t+Cdur_nmda)\n"
  "		  t0=t\n"
  "	      }\n"
  "         }		  \n"
  "	if (flag == 1) { : turn off transmitter, i.e. this synapse enters the offset state	\n"
  "	on_nmda=0\n"
  "    }\n"
  "	         \n"
  "	if (flag == 0) {   : Short term plasticity was implemented(Varela et. al 1997):\n"
  "\n"
  "	rp = unirand()	\n"
  "\n"
  "	D1 = 1 - (1-D1)*exp(-(t - tsyn)/tauD1)\n"
  "	D2 = 1 - (1-D2)*exp(-(t - tsyn)/tauD2)\n"
  "	tsyn = t\n"
  "	\n"
  "	facfactor = F * D1 * D2	\n"
  "	if (F > 3) { \n"
  "	F=3	}\n"
  "	if (facfactor < 0.5) { \n"
  "	facfactor=0.5\n"
  "	}	\n"
  "	D1 = D1 * d1\n"
  "	D2 = D2 * d2\n"
  "	}\n"
  "}\n"
  "\n"
  ":::::::::::: FUNCTIONs and PROCEDUREs ::::::::::::\n"
  "\n"
  "FUNCTION sfunc (v (mV)) {\n"
  "	UNITSOFF\n"
  "	sfunc = 1/(1+0.33*exp(-0.06*v))\n"
  "	UNITSON\n"
  "}\n"
  "\n"
  "FUNCTION eta(Cani (mM)) {\n"
  "	LOCAL taulearn, P1, P2, P4, Cacon\n"
  "	P1 = 0.1\n"
  "	P2 = P1*1e-4\n"
  "	P4 = 1\n"
  "	Cacon = Cani*1e3\n"
  "	taulearn = P1/(P2+Cacon*Cacon*Cacon)+P4\n"
  "	eta = 1/taulearn*0.001\n"
  "}\n"
  "\n"
  "FUNCTION omega(Cani (mM), threshold1 (uM), threshold2 (uM)) {\n"
  "	LOCAL r, mid, Cacon\n"
  "	Cacon = Cani*1e3\n"
  "	r = (threshold2-threshold1)/2\n"
  "	mid = (threshold1+threshold2)/2\n"
  "	if (Cacon <= threshold1) { omega = 0}\n"
  "	else if (Cacon >= threshold2) {	omega = 1/(1+50*exp(-50*(Cacon-threshold2)))}\n"
  "	else {omega = -sqrt(r*r-(Cacon-mid)*(Cacon-mid))}\n"
  "}\n"
  "FUNCTION DA1(DAstart1 (ms), DAstop1 (ms)) {\n"
  "LOCAL DAtemp1, DAtemp2, DAtemp3, DAtemp4, DAtemp5, DAtemp6, DAtemp7, DAtemp8, DAtemp9, DAtemp10, DAtemp11, DAtemp12, DAtemp13, DAtemp14, DAtemp15, DAtemp16, DAtemp17, DAtemp18, DAtemp19, DAtemp20, DAtemp21, DAtemp22, DAtemp23, DAtemp24, DAtemp25, DAtemp26, DAtemp27, DAtemp28, DAtemp29, DAtemp30, DAtemp31, DAtemp32, DAtemp33, DAtemp34,s\n"
  "	DAtemp1 = DAstart1+4000\n"
  "	DAtemp2 = DAtemp1+4000\n"
  "	DAtemp3 = DAtemp2+4000\n"
  "	DAtemp4 = DAtemp3+4000\n"
  "	DAtemp5 = DAtemp4+4000\n"
  "	DAtemp6 = DAtemp5+4000\n"
  "	DAtemp7 = DAtemp6+4000\n"
  "	DAtemp8 = DAtemp7+4000\n"
  "	DAtemp9 = DAtemp8+4000\n"
  "	DAtemp10 = DAtemp9+4000\n"
  "	DAtemp11 = DAtemp10+4000\n"
  "	DAtemp12 = DAtemp11+4000\n"
  "	DAtemp13 = DAtemp12+4000\n"
  "	DAtemp14 = DAtemp13+4000\n"
  "	DAtemp15 = DAtemp14 + 4000 + 100000     : 100sec Gap\n"
  "	DAtemp16 = DAtemp15 + 4000 \n"
  "	DAtemp17 = DAtemp16 + 4000\n"
  "	DAtemp18 = DAtemp17 + 4000\n"
  "	DAtemp19 = DAtemp18 + 4000 \n"
  "	DAtemp20 = DAtemp19 + 4000\n"
  "	DAtemp21 = DAtemp20 + 4000\n"
  "	DAtemp22 = DAtemp21 + 4000 \n"
  "	DAtemp23 = DAtemp22 + 4000\n"
  "	DAtemp24 = DAtemp23 + 4000\n"
  "	DAtemp25 = DAtemp24 + 4000 \n"
  "	DAtemp26 = DAtemp25 + 4000\n"
  "	DAtemp27 = DAtemp26 + 4000\n"
  "	DAtemp28 = DAtemp27 + 4000 \n"
  "	DAtemp29 = DAtemp28 + 4000\n"
  "	DAtemp30 = DAtemp29 + 4000\n"
  "	DAtemp31 = DAtemp30 + 4000 \n"
  "	DAtemp32 = DAtemp31 + 4000\n"
  "	DAtemp33 = DAtemp32 + 4000\n"
  "	DAtemp34 = DAtemp33 + 4000\n"
  "\n"
  "	if (t <= DAstart1) { DA1 = 1.0}\n"
  "	else if (t >= DAstart1 && t <= DAstop1) {DA1 = DA_t1}					: 2nd tone in conditioning\n"
  "		else if (t > DAstop1 && t < DAtemp1) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-DAstop1))}  			: Basal level\n"
  "	else if (t >= DAtemp1 && t <= DAtemp1+500) {DA1=DA_t1}					: 3rd tone\n"
  "		else if (t > DAtemp1+500 && t < DAtemp2) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp1+500)))} 		: Basal level\n"
  "	else if (t >= DAtemp2 && t <= DAtemp2+500) {DA1=DA_t1}					: 4th tone\n"
  "		else if (t > DAtemp2+500 && t < DAtemp3) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp2+500)))} 		: Basal level	\n"
  "	else if (t >= DAtemp3 && t <= DAtemp3+500) {DA1=DA_t1}					: 5th tone\n"
  "		else if (t > DAtemp3+500 && t < DAtemp4) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp3+500)))} 		: Basal level\n"
  "	else if (t >= DAtemp4 && t <= DAtemp4+500) {DA1=DA_t1}					: 6th tone\n"
  "		else if (t > DAtemp4+500 && t < DAtemp5) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp4+500)))} 		: Basal level\n"
  "	else if (t >= DAtemp5 && t <= DAtemp5+500) {DA1=DA_t1}					: 7th tone\n"
  "		else if (t > DAtemp5+500 && t < DAtemp6) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp5+500)))} 		: Basal level\n"
  "	else if (t >= DAtemp6 && t <= DAtemp6+500) {DA1=DA_t1}					: 8th tone\n"
  "		else if (t > DAtemp6+500 && t < DAtemp7) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp6+500)))} 		: Basal level\n"
  "	else if (t >= DAtemp7 && t <= DAtemp7+500) {DA1=DA_t1}					: 9th tone\n"
  "		else if (t > DAtemp7+500 && t < DAtemp8) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp7+500)))} 		: Basal level\n"
  "	else if (t >= DAtemp8 && t <= DAtemp8+500) {DA1=DA_t1}					: 10th tone  \n"
  "		else if (t > DAtemp8+500 && t < DAtemp9) {DA1 = 1.0 + (DA_t1-1)*exp(-Beta1*(t-(DAtemp8+500)))} 		: Basal level\n"
  "	\n"
  "	else if (t >= DAtemp9 && t <= DAtemp9+500) {DA1=DA_t2}					: 11th tone   - Second Step\n"
  "		else if (t > DAtemp9+500 && t < DAtemp10) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp9+500)))}		: Basal level	\n"
  "	else if (t >= DAtemp10 && t <= DAtemp10+500) {DA1=DA_t2}					: 12th tone\n"
  "		else if (t > DAtemp10+500 && t < DAtemp11) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp10+500)))}	: Basal level\n"
  "	else if (t >= DAtemp11 && t <= DAtemp11+500) {DA1=DA_t2}					: 13th tone\n"
  "		else if (t > DAtemp11+500 && t < DAtemp12) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp11+500)))}	: Basal level\n"
  "	else if (t >= DAtemp12 && t <= DAtemp12+500) {DA1=DA_t2}					: 14th tone \n"
  "		else if (t > DAtemp12+500 && t < DAtemp13) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp12+500)))}	: Basal level\n"
  "	else if (t >= DAtemp13 && t <= DAtemp13+500) {DA1=DA_t2}					: 15th tone\n"
  "		else if (t > DAtemp13+500 && t < DAtemp14) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp13+500)))}	: Basal level\n"
  "	else if (t >= DAtemp14 && t <= DAtemp14+500) {DA1=DA_t2}					: 16th tone\n"
  "		else if (t > DAtemp14+500 && t < DAtemp15) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp14+500)))} 	: Basal level\n"
  "	\n"
  "	else if (t >= DAtemp15 && t <= DAtemp15+500) {DA1 = DA_t2}					: 1st tone EE\n"
  "		else if (t > DAtemp15+500 && t < DAtemp16) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp15+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp16 && t <= DAtemp16+500) {DA1 = DA_t2}					: 2nd tone EE\n"
  "		else if (t > DAtemp16+500 && t < DAtemp17) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp16+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp17 && t <= DAtemp17+500) {DA1 = DA_t2}					: 3rd tone EE\n"
  "		else if (t > DAtemp17+500 && t < DAtemp18) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp17+500)))}  	: Basal level	\n"
  "	else if (t >= DAtemp18 && t <= DAtemp18+500) {DA1 = DA_t2}					: 4th tone EE	\n"
  "		else if (t > DAtemp18+500 && t < DAtemp19) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp18+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp19 && t <= DAtemp19+500) {DA1 = DA_t3}					: 5th tone EE\n"
  "		else if (t > DAtemp19+500 && t < DAtemp20) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp19+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp20 && t <= DAtemp20+500) {DA1 = DA_t3}					: 6th tone EE\n"
  "		else if (t > DAtemp20+500 && t < DAtemp21) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp20+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp21 && t <= DAtemp21+500) {DA1 = DA_t3}					: 7th tone EE\n"
  "		else if (t > DAtemp21+500 && t < DAtemp22) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp21+500)))}  	: Basal level	\n"
  "	else if (t >= DAtemp22 && t <= DAtemp22+500) {DA1 = DA_t3}					: 8th tone EE	\n"
  "		else if (t > DAtemp22+500 && t < DAtemp23) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp22+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp23 && t <= DAtemp23+500) {DA1 = DA_t3}					: 9th tone EE\n"
  "		else if (t > DAtemp23+500 && t < DAtemp24) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp23+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp24 && t <= DAtemp24+500) {DA1 = DA_t3}					: 10th tone EE\n"
  "		else if (t > DAtemp24+500 && t < DAtemp25) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp24+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp25 && t <= DAtemp25+500) {DA1 = DA_t3}					: 11th tone EE\n"
  "		else if (t > DAtemp25+500 && t < DAtemp26) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp25+500)))}  	: Basal level	\n"
  "	else if (t >= DAtemp26 && t <= DAtemp26+500) {DA1 = DA_t3}					: 12th tone EE	\n"
  "		else if (t > DAtemp26+500 && t < DAtemp27) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp26+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp27 && t <= DAtemp27+500) {DA1 = DA_t3}					: 13th tone EE\n"
  "		else if (t > DAtemp27+500 && t < DAtemp28) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp27+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp28 && t <= DAtemp28+500) {DA1 = DA_t3}					: 14th tone EE\n"
  "		else if (t > DAtemp28+500 && t < DAtemp29) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp28+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp29 && t <= DAtemp29+500) {DA1 = DA_t3}					: 15th tone EE\n"
  "		else if (t > DAtemp29+500 && t < DAtemp30) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp29+500)))}  	: Basal level	\n"
  "	else if (t >= DAtemp30 && t <= DAtemp30+500) {DA1 = DA_t3}					: 16th tone EE	\n"
  "		else if (t > DAtemp30+500 && t < DAtemp31) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp30+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp31 && t <= DAtemp31+500) {DA1 = DA_t3}					: 17th tone EE\n"
  "		else if (t > DAtemp31+500 && t < DAtemp32) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp31+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp32 && t <= DAtemp32+500) {DA1 = DA_t3}					: 18th tone EE\n"
  "		else if (t > DAtemp32+500 && t < DAtemp33) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp32+500)))}  	: Basal level\n"
  "	else if (t >= DAtemp33 && t <= DAtemp33+500) {DA1 = DA_t3}					: 19th tone EE\n"
  "		else if (t > DAtemp33+500 && t < DAtemp34) {DA1 = 1.0 + (DA_t2-1)*exp(-Beta2*(t-(DAtemp33+500)))}  	: Basal level	\n"
  "	else if (t >= DAtemp34 && t <= DAtemp34+500) {DA1 = DA_t3}					: 20th tone EE		\n"
  "		else  {	DA1 = 1.0}\n"
  "}\n"
  "FUNCTION DA2(DAstart2 (ms), DAstop2 (ms)) {\n"
  "	LOCAL DA2temp1, DA2temp2, DA2temp3, DA2temp4, DA2temp5, DA2temp6, DA2temp7, DA2temp8, DA2temp9, DA2temp10, DA2temp11, DA2temp12, DA2temp13, DA2temp14, DA2temp15, DA2temp16,s\n"
  "	DA2temp1 = DAstart2 + 4000\n"
  "	DA2temp2 = DA2temp1 + 4000\n"
  "	DA2temp3 = DA2temp2 + 4000\n"
  "	DA2temp4 = DA2temp3 + 4000\n"
  "	DA2temp5 = DA2temp4 + 4000\n"
  "	DA2temp6 = DA2temp5 + 4000\n"
  "	DA2temp7 = DA2temp6 + 4000\n"
  "	DA2temp8 = DA2temp7 + 4000\n"
  "	DA2temp9 = DA2temp8 + 4000\n"
  "	DA2temp10 = DA2temp9 + 4000\n"
  "	DA2temp11 = DA2temp10 + 4000\n"
  "	DA2temp12 = DA2temp11 + 4000 \n"
  "	DA2temp13 = DA2temp12 + 4000\n"
  "	DA2temp14 = DA2temp13 + 4000\n"
  "	DA2temp15 = DA2temp14 + 4000\n"
  "	\n"
  "	if (t <= DAstart2) { DA2 = 1.0}\n"
  "	else if (t >= DAstart2 && t <= DAstop2) {DA2 = DA_S }					: 1st shock\n"
  "		else if (t > DAstop2 && t < DA2temp1) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DAstop2+500)))}  					 \n"
  "	else if (t >= DA2temp1 && t <= DA2temp1+100) {DA2=DA_S}					: 2nd shock\n"
  "		else if (t > DA2temp1+100 && t < DA2temp2) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp1+100)))}    				 \n"
  "	else if (t >= DA2temp2 && t <= DA2temp2+100) {DA2=DA_S}					: 3rd shock\n"
  "		else if (t > DA2temp2+100 && t < DA2temp3) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp2+100)))}   				 \n"
  "	else if (t >= DA2temp3 && t <= DA2temp3+100) {DA2=DA_S}					: 4th shock\n"
  "		else if (t > DA2temp3+100 && t < DA2temp4) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp3+100)))}   				 \n"
  "	else if (t >= DA2temp4 && t <= DA2temp4+100) {DA2=DA_S}					: 5th shock\n"
  "		else if (t > DA2temp4+100 && t < DA2temp5) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp4+100)))}   				 \n"
  "	else if (t >= DA2temp5 && t <= DA2temp5+100) {DA2=DA_S}					: 6th shock\n"
  "		else if (t > DA2temp5+100 && t < DA2temp6) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp5+100)))}    				 \n"
  "	else if (t >= DA2temp6 && t <= DA2temp6+100) {DA2=DA_S}					: 7th shock\n"
  "		else if (t > DA2temp6+100 && t < DA2temp7) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp6+100)))}   				 \n"
  "	else if (t >= DA2temp7 && t <= DA2temp7+100) {DA2=DA_S}					: 8th shock\n"
  "		else if (t > DA2temp7+100 && t < DA2temp8) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp7+100)))}   				    \n"
  "	else if (t >= DA2temp8 && t <= DA2temp8+100) {DA2=DA_S }					: 9th shock\n"
  "		else if (t > DA2temp8+100 && t < DA2temp9) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp8+100)))}   				    \n"
  "	else if (t >= DA2temp9 && t <= DA2temp9+100) {DA2=DA_S }					: 10th shock\n"
  "		else if (t > DA2temp9+100 && t < DA2temp10) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp9+100)))}   				    \n"
  "	else if (t >= DA2temp10 && t <= DA2temp10+100) {DA2=DA_S}					: 11th shock\n"
  "		else if (t > DA2temp10+100 && t < DA2temp11) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp10+100)))}   				 \n"
  "	else if (t >= DA2temp11 && t <= DA2temp11+100) {DA2=DA_S }					: 12th shock\n"
  "		else if (t > DA2temp11+100 && t < DA2temp12) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp11+100)))}   				 \n"
  "	else if (t >= DA2temp12 && t <= DA2temp12+100) {DA2=DA_S}					: 13th shock\n"
  "		else if (t > DA2temp12+100 && t < DA2temp13) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp12+100)))}   				 \n"
  "	else if (t >= DA2temp13 && t <= DA2temp13+100) {DA2=DA_S }					: 14th shock\n"
  "		else if (t > DA2temp13+100 && t < DA2temp14) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp13+100)))}   				 \n"
  "	else if (t >= DA2temp14 && t <= DA2temp14+100) {DA2=DA_S}					: 15th shock\n"
  "		else if (t > DA2temp14+100 && t < DA2temp15) {DA2 = 1.0 + (DA_S-1)*exp(-Beta2*(t-(DA2temp14+100)))}   				 \n"
  "	else if (t >= DA2temp15 && t <= DA2temp15+100) {DA2=DA_S}					: 16th shock\n"
  "		else  {	DA2 = 1.0}\n"
  "}\n"
  "FUNCTION unirand() {    : uniform random numbers between 0 and 1\n"
  "        unirand = 0\n"
  "}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
