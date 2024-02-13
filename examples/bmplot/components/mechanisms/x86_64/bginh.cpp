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
 
#define nrn_init _nrn_init__bginh
#define _nrn_initial _nrn_initial__bginh
#define nrn_cur _nrn_cur__bginh
#define _nrn_current _nrn_current__bginh
#define nrn_jacob _nrn_jacob__bginh
#define nrn_state _nrn_state__bginh
#define _net_receive _net_receive__bginh 
#define release release__bginh 
 
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
#define Cdur_gaba _p[3]
#define Cdur_gaba_columnindex 3
#define AlphaTmax_gaba _p[4]
#define AlphaTmax_gaba_columnindex 4
#define Beta_gaba _p[5]
#define Beta_gaba_columnindex 5
#define Erev_gaba _p[6]
#define Erev_gaba_columnindex 6
#define gbar_gaba _p[7]
#define gbar_gaba_columnindex 7
#define Cainf _p[8]
#define Cainf_columnindex 8
#define pooldiam _p[9]
#define pooldiam_columnindex 9
#define z _p[10]
#define z_columnindex 10
#define neuroM _p[11]
#define neuroM_columnindex 11
#define tauCa _p[12]
#define tauCa_columnindex 12
#define P0g _p[13]
#define P0g_columnindex 13
#define fCag _p[14]
#define fCag_columnindex 14
#define lambda1 _p[15]
#define lambda1_columnindex 15
#define lambda2 _p[16]
#define lambda2_columnindex 16
#define threshold1 _p[17]
#define threshold1_columnindex 17
#define threshold2 _p[18]
#define threshold2_columnindex 18
#define initW _p[19]
#define initW_columnindex 19
#define fmax _p[20]
#define fmax_columnindex 20
#define fmin _p[21]
#define fmin_columnindex 21
#define thr_rp _p[22]
#define thr_rp_columnindex 22
#define facfactor _p[23]
#define facfactor_columnindex 23
#define f _p[24]
#define f_columnindex 24
#define tauF _p[25]
#define tauF_columnindex 25
#define d1 _p[26]
#define d1_columnindex 26
#define tauD1 _p[27]
#define tauD1_columnindex 27
#define d2 _p[28]
#define d2_columnindex 28
#define tauD2 _p[29]
#define tauD2_columnindex 29
#define igaba _p[30]
#define igaba_columnindex 30
#define g_gaba _p[31]
#define g_gaba_columnindex 31
#define on_gaba _p[32]
#define on_gaba_columnindex 32
#define limitW _p[33]
#define limitW_columnindex 33
#define ICag _p[34]
#define ICag_columnindex 34
#define Icatotal _p[35]
#define Icatotal_columnindex 35
#define Wmax _p[36]
#define Wmax_columnindex 36
#define Wmin _p[37]
#define Wmin_columnindex 37
#define maxChange _p[38]
#define maxChange_columnindex 38
#define normW _p[39]
#define normW_columnindex 39
#define scaleW _p[40]
#define scaleW_columnindex 40
#define pregid _p[41]
#define pregid_columnindex 41
#define postgid _p[42]
#define postgid_columnindex 42
#define F _p[43]
#define F_columnindex 43
#define D1 _p[44]
#define D1_columnindex 44
#define D2 _p[45]
#define D2_columnindex 45
#define r_nmda _p[46]
#define r_nmda_columnindex 46
#define r_gaba _p[47]
#define r_gaba_columnindex 47
#define capoolcon _p[48]
#define capoolcon_columnindex 48
#define W _p[49]
#define W_columnindex 49
#define eca _p[50]
#define eca_columnindex 50
#define ica _p[51]
#define ica_columnindex 51
#define t0 _p[52]
#define t0_columnindex 52
#define ICan _p[53]
#define ICan_columnindex 53
#define Afactor _p[54]
#define Afactor_columnindex 54
#define dW_gaba _p[55]
#define dW_gaba_columnindex 55
#define rp _p[56]
#define rp_columnindex 56
#define tsyn _p[57]
#define tsyn_columnindex 57
#define fa _p[58]
#define fa_columnindex 58
#define Dr_nmda _p[59]
#define Dr_nmda_columnindex 59
#define Dr_gaba _p[60]
#define Dr_gaba_columnindex 60
#define Dcapoolcon _p[61]
#define Dcapoolcon_columnindex 61
#define DW _p[62]
#define DW_columnindex 62
#define v _p[63]
#define v_columnindex 63
#define _g _p[64]
#define _g_columnindex 64
#define _tsav _p[65]
#define _tsav_columnindex 65
#define _nd_area  *_ppvar[0].get<double*>()
#define _ion_eca	*(_ppvar[2].get<double*>())
#define _ion_ica	*(_ppvar[3].get<double*>())
 
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
 static double _hoc_GAP1(void*);
 static double _hoc_NE2(void*);
 static double _hoc_NEn(void*);
 static double _hoc_eta(void*);
 static double _hoc_omega(void*);
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
 {"GAP1", _hoc_GAP1},
 {"NE2", _hoc_NE2},
 {"NEn", _hoc_NEn},
 {"eta", _hoc_eta},
 {"omega", _hoc_omega},
 {0, 0}
};
#define DA2 DA2_bginh
#define DA1 DA1_bginh
#define GAP1 GAP1_bginh
#define NE2 NE2_bginh
#define NEn NEn_bginh
#define eta eta_bginh
#define omega omega_bginh
 extern double DA2( _threadargsprotocomma_ double , double );
 extern double DA1( _threadargsprotocomma_ double , double );
 extern double GAP1( _threadargsprotocomma_ double , double );
 extern double NE2( _threadargsprotocomma_ double , double );
 extern double NEn( _threadargsprotocomma_ double , double );
 extern double eta( _threadargsprotocomma_ double );
 extern double omega( _threadargsprotocomma_ double , double , double );
 /* declare global and static user variables */
#define Beta2 Beta2_bginh
 double Beta2 = 0.0001;
#define Beta1 Beta1_bginh
 double Beta1 = 0.001;
#define DA_S DA_S_bginh
 double DA_S = 1.6;
#define DA_t3 DA_t3_bginh
 double DA_t3 = 1.25;
#define DA_t2 DA_t2_bginh
 double DA_t2 = 1.5;
#define DA_t1 DA_t1_bginh
 double DA_t1 = 0.7;
#define DAstop2 DAstop2_bginh
 double DAstop2 = 36000;
#define DAstart2 DAstart2_bginh
 double DAstart2 = 35900;
#define DAstop1 DAstop1_bginh
 double DAstop1 = 40000;
#define DAstart1 DAstart1_bginh
 double DAstart1 = 39500;
#define GAPstop1 GAPstop1_bginh
 double GAPstop1 = 196000;
#define GAPstart1 GAPstart1_bginh
 double GAPstart1 = 96000;
#define NE_S NE_S_bginh
 double NE_S = 1;
#define NE_t3 NE_t3_bginh
 double NE_t3 = 1;
#define NE_t2 NE_t2_bginh
 double NE_t2 = 1;
#define NE_t1 NE_t1_bginh
 double NE_t1 = 1;
#define NEstop2 NEstop2_bginh
 double NEstop2 = 36000;
#define NEstart2 NEstart2_bginh
 double NEstart2 = 35900;
#define NEstop1 NEstop1_bginh
 double NEstop1 = 40000;
#define NEstart1 NEstart1_bginh
 double NEstart1 = 39500;
#define k k_bginh
 double k = 0.01;
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
 {"Beta1_bginh", "/ms"},
 {"Beta2_bginh", "/ms"},
 {"srcid", "1"},
 {"destid", "1"},
 {"Cdur_gaba", "ms"},
 {"AlphaTmax_gaba", "/ms"},
 {"Beta_gaba", "/ms"},
 {"Erev_gaba", "mV"},
 {"gbar_gaba", "uS"},
 {"Cainf", "mM"},
 {"pooldiam", "micrometer"},
 {"tauCa", "ms"},
 {"f", "1"},
 {"tauF", "ms"},
 {"d1", "1"},
 {"tauD1", "ms"},
 {"d2", "1"},
 {"tauD2", "ms"},
 {"igaba", "nA"},
 {"g_gaba", "uS"},
 {"ICag", "nA"},
 {"Icatotal", "nA"},
 {0, 0}
};
 static double W0 = 0;
 static double capoolcon0 = 0;
 static double delta_t = 0.01;
 static double r_gaba0 = 0;
 static double r_nmda0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"k_bginh", &k_bginh},
 {"GAPstart1_bginh", &GAPstart1_bginh},
 {"GAPstop1_bginh", &GAPstop1_bginh},
 {"DAstart1_bginh", &DAstart1_bginh},
 {"DAstop1_bginh", &DAstop1_bginh},
 {"DAstart2_bginh", &DAstart2_bginh},
 {"DAstop2_bginh", &DAstop2_bginh},
 {"DA_t1_bginh", &DA_t1_bginh},
 {"DA_t2_bginh", &DA_t2_bginh},
 {"DA_t3_bginh", &DA_t3_bginh},
 {"DA_S_bginh", &DA_S_bginh},
 {"Beta1_bginh", &Beta1_bginh},
 {"Beta2_bginh", &Beta2_bginh},
 {"NEstart1_bginh", &NEstart1_bginh},
 {"NEstop1_bginh", &NEstop1_bginh},
 {"NEstart2_bginh", &NEstart2_bginh},
 {"NEstop2_bginh", &NEstop2_bginh},
 {"NE_t1_bginh", &NE_t1_bginh},
 {"NE_t2_bginh", &NE_t2_bginh},
 {"NE_t3_bginh", &NE_t3_bginh},
 {"NE_S_bginh", &NE_S_bginh},
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
 
#define _cvode_ieq _ppvar[5].literal_value<int>()
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"bginh",
 "srcid",
 "destid",
 "type",
 "Cdur_gaba",
 "AlphaTmax_gaba",
 "Beta_gaba",
 "Erev_gaba",
 "gbar_gaba",
 "Cainf",
 "pooldiam",
 "z",
 "neuroM",
 "tauCa",
 "P0g",
 "fCag",
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
 0,
 "igaba",
 "g_gaba",
 "on_gaba",
 "limitW",
 "ICag",
 "Icatotal",
 "Wmax",
 "Wmin",
 "maxChange",
 "normW",
 "scaleW",
 "pregid",
 "postgid",
 "F",
 "D1",
 "D2",
 0,
 "r_nmda",
 "r_gaba",
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
 	_p = nrn_prop_data_alloc(_mechtype, 66, _prop);
 	/*initialize range parameters*/
 	srcid = -1;
 	destid = -1;
 	type = -1;
 	Cdur_gaba = 0.7254;
 	AlphaTmax_gaba = 1.52;
 	Beta_gaba = 0.014;
 	Erev_gaba = -75;
 	gbar_gaba = 0.0006;
 	Cainf = 5e-05;
 	pooldiam = 1.8172;
 	z = 2;
 	neuroM = 0;
 	tauCa = 50;
 	P0g = 0.01;
 	fCag = 0.024;
 	lambda1 = 1;
 	lambda2 = 0.01;
 	threshold1 = 0.5;
 	threshold2 = 0.6;
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
  }
 	_prop->param = _p;
 	_prop->param_size = 66;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 6, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[2] = &prop_ion->param[0]; /* eca */
 	_ppvar[3] = &prop_ion->param[3]; /* ica */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 
#define _tqitem &(_ppvar[4])
 static void _net_receive(Point_process*, double*, double);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _bginh_reg() {
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
  hoc_register_prop_size(_mechtype, 66, 6);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "netsend");
  hoc_register_dparam_semantics(_mechtype, 5, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 bginh /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/bginh.mod\n");
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
 static int _slist1[3], _dlist1[3];
 static int release(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DW = 1e-12 * limitW * eta ( _threadargscomma_ capoolcon ) * ( lambda1 * omega ( _threadargscomma_ capoolcon , threshold1 , threshold2 ) - lambda2 * GAP1 ( _threadargscomma_ GAPstart1 , GAPstop1 ) * W ) ;
   Dr_gaba = AlphaTmax_gaba * on_gaba * ( 1.0 - r_gaba ) - Beta_gaba * r_gaba ;
   Dcapoolcon = - fCag * Afactor * Icatotal + ( Cainf - capoolcon ) / tauCa ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DW = DW  / (1. - dt*( ( 1e-12 * limitW * eta ( _threadargscomma_ capoolcon ) )*( ( ( - ( lambda2 * GAP1 ( _threadargscomma_ GAPstart1 , GAPstop1 ) )*( 1.0 ) ) ) ) )) ;
 Dr_gaba = Dr_gaba  / (1. - dt*( ( AlphaTmax_gaba * on_gaba )*( ( ( - 1.0 ) ) ) - ( Beta_gaba )*( 1.0 ) )) ;
 Dcapoolcon = Dcapoolcon  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tauCa )) ;
  return 0;
}
 /*END CVODE*/
 static int release (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    W = W + (1. - exp(dt*(( 1e-12 * limitW * eta ( _threadargscomma_ capoolcon ) )*( ( ( - ( lambda2 * GAP1 ( _threadargscomma_ GAPstart1 , GAPstop1 ) )*( 1.0 ) ) ) ))))*(- ( ( ( ( 1e-12 )*( limitW ) )*( eta ( _threadargscomma_ capoolcon ) ) )*( ( ( lambda1 )*( omega ( _threadargscomma_ capoolcon , threshold1 , threshold2 ) ) ) ) ) / ( ( ( ( 1e-12 )*( limitW ) )*( eta ( _threadargscomma_ capoolcon ) ) )*( ( ( - ( ( lambda2 )*( GAP1 ( _threadargscomma_ GAPstart1 , GAPstop1 ) ) )*( 1.0 ) ) ) ) ) - W) ;
    r_gaba = r_gaba + (1. - exp(dt*(( AlphaTmax_gaba * on_gaba )*( ( ( - 1.0 ) ) ) - ( Beta_gaba )*( 1.0 ))))*(- ( ( ( AlphaTmax_gaba )*( on_gaba ) )*( ( 1.0 ) ) ) / ( ( ( AlphaTmax_gaba )*( on_gaba ) )*( ( ( - 1.0 ) ) ) - ( Beta_gaba )*( 1.0 ) ) - r_gaba) ;
    capoolcon = capoolcon + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tauCa)))*(- ( ( ( - fCag )*( Afactor ) )*( Icatotal ) + ( ( Cainf ) ) / tauCa ) / ( ( ( ( - 1.0 ) ) ) / tauCa ) - capoolcon) ;
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
     if ( (  ! on_gaba ) ) {
       t0 = t ;
       on_gaba = 1.0 ;
       net_send ( _tqitem, _args, _pnt, t +  Cdur_gaba , 1.0 ) ;
       }
     else if ( on_gaba  == 1.0 ) {
       net_move ( _tqitem, _pnt, t + Cdur_gaba ) ;
       t0 = t ;
       }
     }
   if ( _lflag  == 1.0 ) {
     on_gaba = 0.0 ;
     }
   if ( _lflag  == 0.0 ) {
     D1 = 1.0 - ( 1.0 - D1 ) * exp ( - ( t - tsyn ) / tauD1 ) ;
     D2 = 1.0 - ( 1.0 - D2 ) * exp ( - ( t - tsyn ) / tauD2 ) ;
     tsyn = t ;
     facfactor = F * D1 * D2 ;
     if ( F > 3.0 ) {
       F = 3.0 ;
       }
     if ( facfactor < 0.7 ) {
       facfactor = 0.7 ;
       }
     D1 = D1 * d1 ;
     D2 = D2 * d2 ;
     }
   } }
 
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
 
double NEn ( _threadargsprotocomma_ double _lNEstart1 , double _lNEstop1 ) {
   double _lNEn;
 double _lNEtemp1 , _lNEtemp2 , _lNEtemp3 , _lNEtemp4 , _lNEtemp5 , _lNEtemp6 , _lNEtemp7 , _lNEtemp8 , _lNEtemp9 , _lNEtemp10 , _lNEtemp11 , _lNEtemp12 , _lNEtemp13 , _lNEtemp14 , _lNEtemp15 , _lNEtemp16 , _lNEtemp17 , _lNEtemp18 , _lNEtemp19 , _lNEtemp20 , _lNEtemp21 , _lNEtemp22 , _lNEtemp23 , _lNEtemp24 , _lNEtemp25 , _lNEtemp26 , _lNEtemp27 , _lNEtemp28 , _lNEtemp29 , _lNEtemp30 , _lNEtemp31 , _lNEtemp32 , _lNEtemp33 , _lNEtemp34 , _ls ;
 _lNEtemp1 = _lNEstart1 + 4000.0 ;
   _lNEtemp2 = _lNEtemp1 + 4000.0 ;
   _lNEtemp3 = _lNEtemp2 + 4000.0 ;
   _lNEtemp4 = _lNEtemp3 + 4000.0 ;
   _lNEtemp5 = _lNEtemp4 + 4000.0 ;
   _lNEtemp6 = _lNEtemp5 + 4000.0 ;
   _lNEtemp7 = _lNEtemp6 + 4000.0 ;
   _lNEtemp8 = _lNEtemp7 + 4000.0 ;
   _lNEtemp9 = _lNEtemp8 + 4000.0 ;
   _lNEtemp10 = _lNEtemp9 + 4000.0 ;
   _lNEtemp11 = _lNEtemp10 + 4000.0 ;
   _lNEtemp12 = _lNEtemp11 + 4000.0 ;
   _lNEtemp13 = _lNEtemp12 + 4000.0 ;
   _lNEtemp14 = _lNEtemp13 + 4000.0 ;
   _lNEtemp15 = _lNEtemp14 + 4000.0 + 100000.0 ;
   _lNEtemp16 = _lNEtemp15 + 4000.0 ;
   _lNEtemp17 = _lNEtemp16 + 4000.0 ;
   _lNEtemp18 = _lNEtemp17 + 4000.0 ;
   _lNEtemp19 = _lNEtemp18 + 4000.0 ;
   _lNEtemp20 = _lNEtemp19 + 4000.0 ;
   _lNEtemp21 = _lNEtemp20 + 4000.0 ;
   _lNEtemp22 = _lNEtemp21 + 4000.0 ;
   _lNEtemp23 = _lNEtemp22 + 4000.0 ;
   _lNEtemp24 = _lNEtemp23 + 4000.0 ;
   _lNEtemp25 = _lNEtemp24 + 4000.0 ;
   _lNEtemp26 = _lNEtemp25 + 4000.0 ;
   _lNEtemp27 = _lNEtemp26 + 4000.0 ;
   _lNEtemp28 = _lNEtemp27 + 4000.0 ;
   _lNEtemp29 = _lNEtemp28 + 4000.0 ;
   _lNEtemp30 = _lNEtemp29 + 4000.0 ;
   _lNEtemp31 = _lNEtemp30 + 4000.0 ;
   _lNEtemp32 = _lNEtemp31 + 4000.0 ;
   _lNEtemp33 = _lNEtemp32 + 4000.0 ;
   _lNEtemp34 = _lNEtemp33 + 4000.0 ;
   if ( t <= _lNEstart1 ) {
     _lNEn = 1.0 ;
     }
   else if ( t >= _lNEstart1  && t <= _lNEstop1 ) {
     _lNEn = NE_t1 ;
     }
   else if ( t > _lNEstop1  && t < _lNEtemp1 ) {
     _lNEn = 1.0 + ( NE_t1 - 1.0 ) * exp ( - Beta1 * ( t - _lNEstop1 ) ) ;
     }
   else if ( t >= _lNEtemp1  && t <= _lNEtemp1 + 500.0 ) {
     _lNEn = NE_t1 ;
     }
   else if ( t > _lNEtemp1 + 500.0  && t < _lNEtemp2 ) {
     _lNEn = 1.0 + ( NE_t1 - 1.0 ) * exp ( - Beta1 * ( t - _lNEstop1 ) ) ;
     }
   else if ( t >= _lNEtemp2  && t <= _lNEtemp2 + 500.0 ) {
     _lNEn = NE_t1 ;
     }
   else if ( t > _lNEtemp2 + 500.0  && t < _lNEtemp3 ) {
     _lNEn = 1.0 + ( NE_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lNEtemp2 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp3  && t <= _lNEtemp3 + 500.0 ) {
     _lNEn = NE_t1 ;
     }
   else if ( t > _lNEtemp3 + 500.0  && t < _lNEtemp4 ) {
     _lNEn = 1.0 + ( NE_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lNEtemp3 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp4  && t <= _lNEtemp4 + 500.0 ) {
     _lNEn = NE_t1 ;
     }
   else if ( t > _lNEtemp4 + 500.0  && t < _lNEtemp5 ) {
     _lNEn = 1.0 + ( NE_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lNEtemp4 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp5  && t <= _lNEtemp5 + 500.0 ) {
     _lNEn = NE_t1 ;
     }
   else if ( t > _lNEtemp5 + 500.0  && t < _lNEtemp6 ) {
     _lNEn = 1.0 + ( NE_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lNEtemp5 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp6  && t <= _lNEtemp6 + 500.0 ) {
     _lNEn = NE_t1 ;
     }
   else if ( t > _lNEtemp6 + 500.0  && t < _lNEtemp7 ) {
     _lNEn = 1.0 + ( NE_t1 - 1.0 ) * exp ( - Beta1 * ( t - ( _lNEtemp6 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp7  && t <= _lNEtemp7 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp7 + 500.0  && t < _lNEtemp8 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp7 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp8  && t <= _lNEtemp8 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp8 + 500.0  && t < _lNEtemp9 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp8 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp9  && t <= _lNEtemp9 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp9 + 500.0  && t < _lNEtemp10 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp9 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp10  && t <= _lNEtemp10 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp10 + 500.0  && t < _lNEtemp11 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp10 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp11  && t <= _lNEtemp11 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp11 + 500.0  && t < _lNEtemp12 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp11 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp12  && t <= _lNEtemp12 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp12 + 500.0  && t < _lNEtemp13 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp12 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp13  && t <= _lNEtemp13 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp13 + 500.0  && t < _lNEtemp14 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp13 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp14  && t <= _lNEtemp14 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp14 + 500.0  && t < _lNEtemp15 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp14 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp15  && t <= _lNEtemp15 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp15 + 500.0  && t < _lNEtemp16 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp15 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp16  && t <= _lNEtemp16 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp16 + 500.0  && t < _lNEtemp17 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp16 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp17  && t <= _lNEtemp17 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp17 + 500.0  && t < _lNEtemp18 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp17 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp18  && t <= _lNEtemp18 + 500.0 ) {
     _lNEn = NE_t2 ;
     }
   else if ( t > _lNEtemp18 + 500.0  && t < _lNEtemp19 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp18 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp19  && t <= _lNEtemp19 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp19 + 500.0  && t < _lNEtemp20 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp19 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp20  && t <= _lNEtemp20 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp20 + 500.0  && t < _lNEtemp21 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp20 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp21  && t <= _lNEtemp21 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp21 + 500.0  && t < _lNEtemp22 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp21 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp22  && t <= _lNEtemp22 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp22 + 500.0  && t < _lNEtemp23 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp22 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp23  && t <= _lNEtemp23 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp23 + 500.0  && t < _lNEtemp24 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp23 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp24  && t <= _lNEtemp24 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp24 + 500.0  && t < _lNEtemp25 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp24 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp25  && t <= _lNEtemp25 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp25 + 500.0  && t < _lNEtemp26 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp25 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp26  && t <= _lNEtemp26 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp26 + 500.0  && t < _lNEtemp27 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp26 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp27  && t <= _lNEtemp27 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp27 + 500.0  && t < _lNEtemp28 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp27 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp28  && t <= _lNEtemp28 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp28 + 500.0  && t < _lNEtemp29 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp28 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp29  && t <= _lNEtemp29 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp29 + 500.0  && t < _lNEtemp30 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp29 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp30  && t <= _lNEtemp30 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp30 + 500.0  && t < _lNEtemp31 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp30 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp31  && t <= _lNEtemp31 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp31 + 500.0  && t < _lNEtemp32 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp31 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp32  && t <= _lNEtemp32 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp32 + 500.0  && t < _lNEtemp33 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp32 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp33  && t <= _lNEtemp33 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else if ( t > _lNEtemp33 + 500.0  && t < _lNEtemp34 ) {
     _lNEn = 1.0 + ( NE_t2 - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEtemp33 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNEtemp34  && t <= _lNEtemp34 + 500.0 ) {
     _lNEn = NE_t3 ;
     }
   else {
     _lNEn = 1.0 ;
     }
   
return _lNEn;
 }
 
static double _hoc_NEn(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  NEn ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}
 
double NE2 ( _threadargsprotocomma_ double _lNEstart2 , double _lNEstop2 ) {
   double _lNE2;
 double _lNE2temp1 , _lNE2temp2 , _lNE2temp3 , _lNE2temp4 , _lNE2temp5 , _lNE2temp6 , _lNE2temp7 , _lNE2temp8 , _lNE2temp9 , _lNE2temp10 , _lNE2temp11 , _lNE2temp12 , _lNE2temp13 , _lNE2temp14 , _lNE2temp15 , _lNE2temp16 , _ls ;
 _lNE2temp1 = _lNEstart2 + 4000.0 ;
   _lNE2temp2 = _lNE2temp1 + 4000.0 ;
   _lNE2temp3 = _lNE2temp2 + 4000.0 ;
   _lNE2temp4 = _lNE2temp3 + 4000.0 ;
   _lNE2temp5 = _lNE2temp4 + 4000.0 ;
   _lNE2temp6 = _lNE2temp5 + 4000.0 ;
   _lNE2temp7 = _lNE2temp6 + 4000.0 ;
   _lNE2temp8 = _lNE2temp7 + 4000.0 ;
   _lNE2temp9 = _lNE2temp8 + 4000.0 ;
   _lNE2temp10 = _lNE2temp9 + 4000.0 ;
   _lNE2temp11 = _lNE2temp10 + 4000.0 ;
   _lNE2temp12 = _lNE2temp11 + 4000.0 ;
   _lNE2temp13 = _lNE2temp12 + 4000.0 ;
   _lNE2temp14 = _lNE2temp13 + 4000.0 ;
   _lNE2temp15 = _lNE2temp14 + 4000.0 ;
   if ( t <= _lNEstart2 ) {
     _lNE2 = 1.0 ;
     }
   else if ( t >= _lNEstart2  && t <= _lNEstop2 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNEstop2  && t < _lNE2temp1 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNEstop2 + 500.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp1  && t <= _lNE2temp1 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp1 + 100.0  && t < _lNE2temp2 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp1 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp2  && t <= _lNE2temp2 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp2 + 100.0  && t < _lNE2temp3 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp2 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp3  && t <= _lNE2temp3 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp3 + 100.0  && t < _lNE2temp4 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp3 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp4  && t <= _lNE2temp4 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp4 + 100.0  && t < _lNE2temp5 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp4 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp5  && t <= _lNE2temp5 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp5 + 100.0  && t < _lNE2temp6 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp5 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp6  && t <= _lNE2temp6 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp6 + 100.0  && t < _lNE2temp7 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp6 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp7  && t <= _lNE2temp7 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp7 + 100.0  && t < _lNE2temp8 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp7 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp8  && t <= _lNE2temp8 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp8 + 100.0  && t < _lNE2temp9 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp8 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp9  && t <= _lNE2temp9 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp9 + 100.0  && t < _lNE2temp10 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp9 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp10  && t <= _lNE2temp10 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp10 + 100.0  && t < _lNE2temp11 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp10 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp11  && t <= _lNE2temp11 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp11 + 100.0  && t < _lNE2temp12 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp11 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp12  && t <= _lNE2temp12 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp12 + 100.0  && t < _lNE2temp13 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp12 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp13  && t <= _lNE2temp13 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp13 + 100.0  && t < _lNE2temp14 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp13 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp14  && t <= _lNE2temp14 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else if ( t > _lNE2temp14 + 100.0  && t < _lNE2temp15 ) {
     _lNE2 = 1.0 + ( NE_S - 1.0 ) * exp ( - Beta2 * ( t - ( _lNE2temp14 + 100.0 ) ) ) ;
     }
   else if ( t >= _lNE2temp15  && t <= _lNE2temp15 + 100.0 ) {
     _lNE2 = NE_S ;
     }
   else {
     _lNE2 = 1.0 ;
     }
   
return _lNE2;
 }
 
static double _hoc_NE2(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  NE2 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}
 
double GAP1 ( _threadargsprotocomma_ double _lGAPstart1 , double _lGAPstop1 ) {
   double _lGAP1;
 double _ls ;
 if ( t <= _lGAPstart1 ) {
     _lGAP1 = 1.0 ;
     }
   else if ( t >= _lGAPstop1 ) {
     _lGAP1 = 1.0 ;
     }
   else {
     _lGAP1 = 1.0 ;
     }
   
return _lGAP1;
 }
 
static double _hoc_GAP1(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  GAP1 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
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
  ica = _ion_ica;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
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
  ica = _ion_ica;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 0);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 3);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  W = W0;
  capoolcon = capoolcon0;
  r_gaba = r_gaba0;
  r_nmda = r_nmda0;
 {
   on_gaba = 0.0 ;
   r_gaba = 0.0 ;
   W = initW ;
   limitW = 1.0 ;
   t0 = - 1.0 ;
   Wmax = fmax * initW ;
   Wmin = fmin * initW ;
   maxChange = ( Wmax - Wmin ) / 10.0 ;
   dW_gaba = 0.0 ;
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
  ica = _ion_ica;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   if ( ( eta ( _threadargscomma_ capoolcon ) * ( lambda1 * omega ( _threadargscomma_ capoolcon , threshold1 , threshold2 ) - lambda2 * GAP1 ( _threadargscomma_ GAPstart1 , GAPstop1 ) * W ) ) > 0.0  && W >= Wmax ) {
     limitW = 1e-12 ;
     }
   else if ( ( eta ( _threadargscomma_ capoolcon ) * ( lambda1 * omega ( _threadargscomma_ capoolcon , threshold1 , threshold2 ) - lambda2 * GAP1 ( _threadargscomma_ GAPstart1 , GAPstop1 ) * W ) ) < 0.0  && W <= Wmin ) {
     limitW = 1e-12 ;
     }
   else {
     limitW = 1.0 ;
     }
   if ( neuroM  == 1.0 ) {
     g_gaba = gbar_gaba * r_gaba * facfactor * DA1 ( _threadargscomma_ DAstart1 , DAstop1 ) * DA2 ( _threadargscomma_ DAstart2 , DAstop2 ) ;
     }
   else if ( neuroM  == 2.0 ) {
     g_gaba = gbar_gaba * r_gaba * facfactor * NEn ( _threadargscomma_ NEstart1 , NEstop1 ) * NE2 ( _threadargscomma_ NEstart2 , NEstop2 ) ;
     }
   else if ( neuroM  == 3.0 ) {
     g_gaba = gbar_gaba * r_gaba * facfactor * DA1 ( _threadargscomma_ DAstart1 , DAstop1 ) * DA2 ( _threadargscomma_ DAstart2 , DAstop2 ) * NEn ( _threadargscomma_ NEstart1 , NEstop1 ) * NE2 ( _threadargscomma_ NEstart2 , NEstop2 ) ;
     }
   else {
     g_gaba = gbar_gaba * r_gaba * facfactor ;
     }
   igaba = W * g_gaba * ( v - Erev_gaba ) ;
   ICag = P0g * g_gaba * ( v - eca ) ;
   Icatotal = ICag + k * ica * 4.0 * pilocal * ( pow( ( 15.0 / 2.0 ) , 2.0 ) ) * ( 0.01 ) ;
   }
 _current += igaba;

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
  ica = _ion_ica;
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
  ica = _ion_ica;
 {   release(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = W_columnindex;  _dlist1[0] = DW_columnindex;
 _slist1[1] = r_gaba_columnindex;  _dlist1[1] = Dr_gaba_columnindex;
 _slist1[2] = capoolcon_columnindex;  _dlist1[2] = Dcapoolcon_columnindex;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/bginh.mod";
    const char* nmodl_file_text = 
  ":Interneuron Cells to Pyramidal Cells GABA with local Ca2+ pool and read public soma Ca2+ pool\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS bginh \n"
  "	USEION ca READ eca,ica\n"
  "	NONSPECIFIC_CURRENT igaba\n"
  "	RANGE initW\n"
  "	RANGE Cdur_gaba, AlphaTmax_gaba, Beta_gaba, Erev_gaba, gbar_gaba, W, on_gaba, g_gaba\n"
  "	RANGE eca, tauCa, Icatotal\n"
  "	RANGE ICag, P0g, fCag\n"
  "	RANGE Cainf, pooldiam, z\n"
  "	RANGE lambda1, lambda2, threshold1, threshold2\n"
  "	RANGE fmax, fmin, Wmax, Wmin, maxChange, normW, scaleW, srcid, destid,limitW\n"
  "	RANGE pregid,postgid, thr_rp\n"
  "	RANGE F, f, tauF, D1, d1, tauD1, D2, d2, tauD2\n"
  "	RANGE facfactor\n"
  "    RANGE neuroM,type\n"
  "}\n"
  "\n"
  "UNITS {\n"
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
  "	Cdur_gaba = 0.7254 (ms)\n"
  "	AlphaTmax_gaba = 1.52 (/ms):7.2609 (/ms): 2.2609 (/ms): 3.2609 (/ms)   : 7.2609 as original\n"
  "	Beta_gaba = 0.014(/ms) : 0.147 (/ms) : 0.2667 (/ms):         : 0.2667 as original\n"
  "	Erev_gaba = -75 (mV) : -75 as original\n"
  "	gbar_gaba = 0.6e-3 (uS)\n"
  "\n"
  "	Cainf = 50e-6 (mM)\n"
  "	pooldiam =  1.8172 (micrometer)\n"
  "	z = 2\n"
  "\n"
  "    neuroM = 0\n"
  "	k = 0.01	\n"
  "	\n"
  "	tauCa = 50 (ms)\n"
  "	\n"
  "	P0g = .01\n"
  "	fCag = .024\n"
  "	\n"
  "	lambda1 = 1 : 0.7 : 0.6 : 0.7 : 1.0 : 0.5 : 1.5 :3 : 4 : 3 : 2 : 3.0 : 2.0\n"
  "	lambda2 = .01\n"
  "	threshold1 = 0.5  :0.47 :  0.48 : 0.45 : 0.4 : 0.95 : 1.35 :0.75 :0.55 (uM)\n"
  "	threshold2 = 0.6 :0.52 :  0.53 : 0.5 : 0.45 : 1.0 : 1.4 : 0.8 : 0.65 :0.70 (uM)\n"
  "\n"
  "	:GABA Weight\n"
  "	initW = 5.0 : 3.0 : 4.0 : 5.0 : 4.2 : 3.5 :4.5 : :  :  3 :  2.5 : 5\n"
  "	fmax = 3 : 2.85 :4 : 3 : 3\n"
  "	fmin = .8\n"
  "	\n"
  "	GAPstart1 = 96000 \n"
  "	GAPstop1 = 196000\n"
  "	\n"
  "	thr_rp = 1 : .7\n"
  "	\n"
  "	facfactor = 1\n"
  "	: the (1) is needed for the range limits to be effective\n"
  "        f = 0 (1) < 0, 1e9 > : 1.3 (1) < 0, 1e9 >    : facilitation\n"
  "        tauF = 20 (ms) < 1e-9, 1e9 >\n"
  "        d1 = 0.95 (1) < 0, 1 >     : fast depression\n"
  "        tauD1 = 40 (ms) < 1e-9, 1e9 >\n"
  "        d2 = 0.9 (1) < 0, 1 >     : slow depression\n"
  "        tauD2 = 70 (ms) < 1e-9, 1e9 >	\n"
  "	\n"
  "    DAstart1 = 39500\n"
  "	DAstop1 = 40000	\n"
  "	DAstart2 = 35900\n"
  "	DAstop2 = 36000	\n"
  "\n"
  "	DA_t1 = 0.7 : 0.7\n"
  "	DA_t2 = 1.5 : 1.3 : 1.2\n"
  "	DA_t3 = 1.25\n"
  "	DA_S = 1.6 : 1.8 : 1.8					\n"
  "	Beta1 = 0.001  (/ms) : 1/decay time for neuromodulators\n"
  "	Beta2 = 0.0001  (/ms)	\n"
  "	\n"
  "	NEstart1 = 39500\n"
  "	NEstop1 = 40000	\n"
  "	NEstart2 = 35900\n"
  "	NEstop2 = 36000		\n"
  "\n"
  "\n"
  "	NE_t1 = 1 : 1 : 0.95\n"
  "\n"
  "	NE_t2 = 1 :1 : 0.7 : 0.8\n"
  "	NE_t3 = 1\n"
  "	NE_S = 1 : 0.4\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)\n"
  "	eca (mV)\n"
  "	ica (nA)\n"
  "	\n"
  "	igaba (nA)\n"
  "	g_gaba (uS)\n"
  "	on_gaba\n"
  "\n"
  "	limitW\n"
  "\n"
  "	t0 (ms)\n"
  "\n"
  "	ICan (nA)\n"
  "	ICag (nA)\n"
  "	Afactor	(mM/ms/nA)\n"
  "	Icatotal (nA)\n"
  "\n"
  "	dW_gaba\n"
  "	Wmax\n"
  "	Wmin\n"
  "	maxChange\n"
  "	normW\n"
  "	scaleW\n"
  "	\n"
  "	pregid\n"
  "	postgid\n"
  "\n"
  "	rp\n"
  "	tsyn\n"
  "	\n"
  "	fa\n"
  "	F\n"
  "	D1\n"
  "	D2	\n"
  "}\n"
  "\n"
  "STATE { r_nmda r_gaba capoolcon W }\n"
  "\n"
  "INITIAL {\n"
  "\n"
  "	on_gaba = 0\n"
  "	r_gaba = 0\n"
  "	W = initW\n"
  "	limitW = 1\n"
  "\n"
  "	t0 = -1\n"
  "\n"
  "	Wmax = fmax*initW\n"
  "	Wmin = fmin*initW\n"
  "	maxChange = (Wmax-Wmin)/10\n"
  "	dW_gaba = 0\n"
  "\n"
  "	capoolcon = Cainf\n"
  "	Afactor	= 1/(z*FARADAY*4/3*pilocal*(pooldiam/2)^3)*(1e6)\n"
  "\n"
  "	fa =0\n"
  "	F = 1\n"
  "	D1 = 1\n"
  "	D2 = 1	\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*GAP1(GAPstart1, GAPstop1)*W))>0&&W>=Wmax) {\n"
  "        limitW=1e-12\n"
  "	} else if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*GAP1(GAPstart1, GAPstop1)*W))<0&&W<=Wmin) {\n"
  "        limitW=1e-12\n"
  "	} else {\n"
  "	limitW=1 }\n"
  "	\n"
  "	SOLVE release METHOD cnexp\n"
  "	 : if (W >= Wmax || W <= Wmin ) {     : for limiting the weight\n"
  "	 : limitW=1e-12\n"
  "	 : } else {\n"
  "	  : limitW=1\n"
  "	 : }\n"
  "	 :if (W > Wmax) { \n"
  "		:W = Wmax\n"
  "	:} else if (W < Wmin) {\n"
  " 		:W = Wmin\n"
  "	:}\n"
  "	 \n"
  "	    if (neuroM==1) {\n"
  "	g_gaba = gbar_gaba*r_gaba*facfactor*DA1(DAstart1,DAstop1)*DA2(DAstart2,DAstop2)   : Dopamine effect on GABA	\n"
  "	} else if (neuroM==2) {\n"
  "	g_gaba = gbar_gaba*r_gaba*facfactor*NEn(NEstart1,NEstop1)*NE2(NEstart2,NEstop2)   : Norepilocalnephrine effect on GABA		    	\n"
  "	} else if (neuroM==3) {\n"
  "	g_gaba = gbar_gaba*r_gaba*facfactor*DA1(DAstart1,DAstop1)*DA2(DAstart2,DAstop2)*NEn(NEstart1,NEstop1)*NE2(NEstart2,NEstop2)   : Dopamine & Norepilocalnephrine effect on GABA		    \n"
  "	} else {\n"
  "	g_gaba = gbar_gaba*r_gaba*facfactor\n"
  "	}\n"
  "\n"
  "    igaba = W*g_gaba*(v - Erev_gaba)\n"
  "\n"
  "	ICag = P0g*g_gaba*(v - eca)	\n"
  "	Icatotal = ICag + k*ica*4*pilocal*((15/2)^2)*(0.01)    :  icag+k*ica*Area of soma*unit change\n"
  "\n"
  "	\n"
  "}\n"
  "\n"
  "DERIVATIVE release {\n"
  "    \n"
  "	: W' = eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*GAP1(GAPstart1, GAPstop1)*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)\n"
  "    \n"
  "	\n"
  "	W' = 1e-12*limitW*eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*GAP1(GAPstart1, GAPstop1)*W)\n"
  "	\n"
  "	r_gaba' = AlphaTmax_gaba*on_gaba*(1-r_gaba)-Beta_gaba*r_gaba\n"
  "    capoolcon'= -fCag*Afactor*Icatotal + (Cainf-capoolcon)/tauCa		\n"
  "}\n"
  "\n"
  "NET_RECEIVE(dummy_weight) {\n"
  "      if (flag==0) {           :a spilocalke arrived, start onset state if not already on\n"
  "         if ((!on_gaba)){       :this synpase joins the set of synapses in onset state\n"
  "           t0=t\n"
  "	      on_gaba=1		\n"
  "	      net_send(Cdur_gaba,1)  \n"
  "         } else if (on_gaba==1) {             :already in onset state, so move offset time\n"
  "          net_move(t+Cdur_gaba)\n"
  "		  t0=t\n"
  "	      }\n"
  "         }		  \n"
  "	if (flag == 1) { : turn off transmitter, i.e. this synapse enters the offset state	\n"
  "	on_gaba=0\n"
  "    }\n"
  "	\n"
  "if (flag == 0) {  : Short term plasticity was implemented(Varela et. al 1997):\n"
  "	:rp = unirand()	\n"
  "	\n"
  "	:F  = 1 + (F-1)* exp(-(t - tsyn)/tauF)\n"
  "	D1 = 1 - (1-D1)*exp(-(t - tsyn)/tauD1)\n"
  "	D2 = 1 - (1-D2)*exp(-(t - tsyn)/tauD2)\n"
  " :printf(\"%g\\t%g\\t%g\\t%g\\t%g\\t%g\\n\", t, t-tsyn, F, D1, D2, facfactor)\n"
  "	:printf(\"%g\\t%g\\t%g\\t%g\\n\", F, D1, D2, facfactor)\n"
  "	tsyn = t\n"
  "	\n"
  "	facfactor = F * D1 * D2\n"
  "\n"
  "	::F = F+f  :F * f\n"
  "	\n"
  "	if (F > 3) { \n"
  "	F=3	}	\n"
  "	if (facfactor < 0.7) { \n"
  "	facfactor=0.7\n"
  "	}\n"
  "	D1 = D1 * d1\n"
  "	D2 = D2 * d2\n"
  ":printf(\"\\t%g\\t%g\\t%g\\n\", F, D1, D2)\n"
  "}\n"
  "}\n"
  "\n"
  ":::::::::::: FUNCTIONs and PROCEDUREs ::::::::::::\n"
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
  "	LOCAL DAtemp1, DAtemp2, DAtemp3, DAtemp4, DAtemp5, DAtemp6, DAtemp7, DAtemp8, DAtemp9, DAtemp10, DAtemp11, DAtemp12, DAtemp13, DAtemp14, DAtemp15, DAtemp16, DAtemp17, DAtemp18, DAtemp19, DAtemp20, DAtemp21, DAtemp22, DAtemp23, DAtemp24, DAtemp25, DAtemp26, DAtemp27, DAtemp28, DAtemp29, DAtemp30, DAtemp31, DAtemp32, DAtemp33, DAtemp34,s\n"
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
  "\n"
  "FUNCTION NEn(NEstart1 (ms), NEstop1 (ms)) {\n"
  "	LOCAL NEtemp1, NEtemp2, NEtemp3, NEtemp4, NEtemp5, NEtemp6, NEtemp7, NEtemp8, NEtemp9, NEtemp10, NEtemp11, NEtemp12, NEtemp13, NEtemp14, NEtemp15, NEtemp16, NEtemp17, NEtemp18, NEtemp19, NEtemp20, NEtemp21, NEtemp22, NEtemp23, NEtemp24, NEtemp25, NEtemp26, NEtemp27, NEtemp28, NEtemp29, NEtemp30, NEtemp31, NEtemp32, NEtemp33, NEtemp34,s\n"
  "	NEtemp1 = NEstart1+4000\n"
  "	NEtemp2 = NEtemp1+4000\n"
  "	NEtemp3 = NEtemp2+4000\n"
  "	NEtemp4 = NEtemp3+4000\n"
  "	NEtemp5 = NEtemp4+4000\n"
  "	NEtemp6 = NEtemp5+4000\n"
  "	NEtemp7 = NEtemp6+4000\n"
  "	NEtemp8 = NEtemp7+4000\n"
  "	NEtemp9 = NEtemp8+4000\n"
  "	NEtemp10 = NEtemp9+4000\n"
  "	NEtemp11 = NEtemp10+4000\n"
  "	NEtemp12 = NEtemp11+4000\n"
  "	NEtemp13 = NEtemp12+4000\n"
  "	NEtemp14 = NEtemp13+4000\n"
  "	NEtemp15 = NEtemp14 + 4000 + 100000     : 100sec Gap\n"
  "	NEtemp16 = NEtemp15 + 4000 \n"
  "	NEtemp17 = NEtemp16 + 4000\n"
  "	NEtemp18 = NEtemp17 + 4000\n"
  "	NEtemp19 = NEtemp18 + 4000 \n"
  "	NEtemp20 = NEtemp19 + 4000\n"
  "	NEtemp21 = NEtemp20 + 4000\n"
  "	NEtemp22 = NEtemp21 + 4000 \n"
  "	NEtemp23 = NEtemp22 + 4000\n"
  "	NEtemp24 = NEtemp23 + 4000\n"
  "	NEtemp25 = NEtemp24 + 4000 \n"
  "	NEtemp26 = NEtemp25 + 4000\n"
  "	NEtemp27 = NEtemp26 + 4000\n"
  "	NEtemp28 = NEtemp27 + 4000 \n"
  "	NEtemp29 = NEtemp28 + 4000\n"
  "	NEtemp30 = NEtemp29 + 4000\n"
  "	NEtemp31 = NEtemp30 + 4000 \n"
  "	NEtemp32 = NEtemp31 + 4000\n"
  "	NEtemp33 = NEtemp32 + 4000\n"
  "	NEtemp34 = NEtemp33 + 4000\n"
  "\n"
  "	if (t <= NEstart1) { NEn = 1.0}\n"
  "	else if (t >= NEstart1 && t <= NEstop1) {NEn = NE_t1}					: 2nd tone in early conditioning (EC)\n"
  "		else if (t > NEstop1 && t < NEtemp1) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-NEstop1))}  		: Basal level\n"
  "	else if (t >= NEtemp1 && t <= NEtemp1+500) {NEn = NE_t1}					: 3rd tone EC\n"
  "		else if (t > NEtemp1+500 && t < NEtemp2) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-NEstop1))}  	: Basal level\n"
  "	else if (t >= NEtemp2 && t <= NEtemp2+500) {NEn = NE_t1}					: 4th tone EC\n"
  "		else if (t > NEtemp2+500 && t < NEtemp3) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp2+500)))}  	: Basal level	\n"
  "	else if (t >= NEtemp3 && t <= NEtemp3+500) {NEn = NE_t1}					: 5th tone EC\n"
  "		else if (t > NEtemp3+500 && t < NEtemp4) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp3+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp4 && t <= NEtemp4+500) {NEn = NE_t1}					: 6th tone EC\n"
  "		else if (t > NEtemp4+500 && t < NEtemp5) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp4+500)))}  		: Basal level\n"
  "	else if (t >= NEtemp5 && t <= NEtemp5+500) {NEn = NE_t1}					: 7th tone EC\n"
  "		else if (t > NEtemp5+500 && t < NEtemp6) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp5+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp6 && t <= NEtemp6+500) {NEn = NE_t1}					: 8th tone EC\n"
  "		else if (t > NEtemp6+500 && t < NEtemp7) {NEn = 1.0 + (NE_t1-1)*exp(-Beta1*(t-(NEtemp6+500)))}  	: Basal level\n"
  "	\n"
  "	else if (t >= NEtemp7 && t <= NEtemp7+500) {NEn = NE_t2}					: 9th tone	- Second Step late cond (LC)\n"
  "		else if (t > NEtemp7+500 && t < NEtemp8) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp7+500)))}  		: Basal level\n"
  "	else if (t >= NEtemp8 && t <= NEtemp8+500) {NEn = NE_t2}					: 10th tone  LC\n"
  "		else if (t > NEtemp8+500 && t < NEtemp9) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp8+500)))}		: Basal level	\n"
  "	else if (t >= NEtemp9 && t <= NEtemp9+500) {NEn = NE_t2}					: 11th tone  LC \n"
  "		else if (t > NEtemp9+500 && t < NEtemp10) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp9+500)))}  	: Basal level	\n"
  "	else if (t >= NEtemp10 && t <= NEtemp10+500) {NEn = NE_t2}					: 12th tone  LC\n"
  "		else if (t > NEtemp10+500 && t < NEtemp11) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp10+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp11 && t <= NEtemp11+500) {NEn = NE_t2}					: 13th tone  LC\n"
  "		else if (t > NEtemp11+500 && t < NEtemp12) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp11+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp12 && t <= NEtemp12+500) {NEn = NE_t2}					: 14th tone  LC\n"
  "		else if (t > NEtemp12+500 && t < NEtemp13) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp12+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp13 && t <= NEtemp13+500) {NEn = NE_t2}					: 15th tone  LC\n"
  "		else if (t > NEtemp13+500 && t < NEtemp14) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp13+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp14 && t <= NEtemp14+500) {NEn = NE_t2}					: 16th tone  LC\n"
  "		else if (t > NEtemp14+500 && t < NEtemp15) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp14+500)))}  	: Basal level\n"
  "	\n"
  "	else if (t >= NEtemp15 && t <= NEtemp15+500) {NEn = NE_t2}					: 1st tone EE\n"
  "		else if (t > NEtemp15+500 && t < NEtemp16) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp15+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp16 && t <= NEtemp16+500) {NEn = NE_t2}					: 2nd tone EE\n"
  "		else if (t > NEtemp16+500 && t < NEtemp17) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp16+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp17 && t <= NEtemp17+500) {NEn = NE_t2}					: 3rd tone EE\n"
  "		else if (t > NEtemp17+500 && t < NEtemp18) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp17+500)))}  	: Basal level	\n"
  "	else if (t >= NEtemp18 && t <= NEtemp18+500) {NEn = NE_t2}					: 4th tone EE	\n"
  "		else if (t > NEtemp18+500 && t < NEtemp19) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp18+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp19 && t <= NEtemp19+500) {NEn = NE_t3}					: 5th tone EE\n"
  "		else if (t > NEtemp19+500 && t < NEtemp20) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp19+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp20 && t <= NEtemp20+500) {NEn = NE_t3}					: 6th tone EE\n"
  "		else if (t > NEtemp20+500 && t < NEtemp21) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp20+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp21 && t <= NEtemp21+500) {NEn = NE_t3}					: 7th tone EE\n"
  "		else if (t > NEtemp21+500 && t < NEtemp22) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp21+500)))}  	: Basal level	\n"
  "	else if (t >= NEtemp22 && t <= NEtemp22+500) {NEn = NE_t3}					: 8th tone EE	\n"
  "		else if (t > NEtemp22+500 && t < NEtemp23) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp22+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp23 && t <= NEtemp23+500) {NEn = NE_t3}					: 9th tone EE\n"
  "		else if (t > NEtemp23+500 && t < NEtemp24) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp23+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp24 && t <= NEtemp24+500) {NEn = NE_t3}					: 10th tone EE\n"
  "		else if (t > NEtemp24+500 && t < NEtemp25) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp24+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp25 && t <= NEtemp25+500) {NEn = NE_t3}					: 11th tone EE\n"
  "		else if (t > NEtemp25+500 && t < NEtemp26) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp25+500)))}  	: Basal level	\n"
  "	else if (t >= NEtemp26 && t <= NEtemp26+500) {NEn = NE_t3}					: 12th tone EE	\n"
  "		else if (t > NEtemp26+500 && t < NEtemp27) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp26+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp27 && t <= NEtemp27+500) {NEn = NE_t3}					: 13th tone EE\n"
  "		else if (t > NEtemp27+500 && t < NEtemp28) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp27+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp28 && t <= NEtemp28+500) {NEn = NE_t3}					: 14th tone EE\n"
  "		else if (t > NEtemp28+500 && t < NEtemp29) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp28+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp29 && t <= NEtemp29+500) {NEn = NE_t3}					: 15th tone EE\n"
  "		else if (t > NEtemp29+500 && t < NEtemp30) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp29+500)))}  	: Basal level	\n"
  "	else if (t >= NEtemp30 && t <= NEtemp30+500) {NEn = NE_t3}					: 16th tone EE	\n"
  "		else if (t > NEtemp30+500 && t < NEtemp31) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp30+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp31 && t <= NEtemp31+500) {NEn = NE_t3}					: 17th tone EE\n"
  "		else if (t > NEtemp31+500 && t < NEtemp32) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp31+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp32 && t <= NEtemp32+500) {NEn = NE_t3}					: 18th tone EE\n"
  "		else if (t > NEtemp32+500 && t < NEtemp33) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp32+500)))}  	: Basal level\n"
  "	else if (t >= NEtemp33 && t <= NEtemp33+500) {NEn = NE_t3}					: 19th tone EE\n"
  "		else if (t > NEtemp33+500 && t < NEtemp34) {NEn = 1.0 + (NE_t2-1)*exp(-Beta2*(t-(NEtemp33+500)))}  	: Basal level	\n"
  "	else if (t >= NEtemp34 && t <= NEtemp34+500) {NEn = NE_t3}					: 20th tone EE		\n"
  "		else  {	NEn = 1.0}\n"
  "\n"
  "}\n"
  "FUNCTION NE2(NEstart2 (ms), NEstop2 (ms)) {\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "	LOCAL NE2temp1, NE2temp2, NE2temp3, NE2temp4, NE2temp5, NE2temp6, NE2temp7, NE2temp8, NE2temp9, NE2temp10, NE2temp11, NE2temp12, NE2temp13, NE2temp14, NE2temp15, NE2temp16,s\n"
  "	NE2temp1 = NEstart2 + 4000\n"
  "	NE2temp2 = NE2temp1 + 4000\n"
  "	NE2temp3 = NE2temp2 + 4000\n"
  "	NE2temp4 = NE2temp3 + 4000\n"
  "	NE2temp5 = NE2temp4 + 4000\n"
  "	NE2temp6 = NE2temp5 + 4000\n"
  "	NE2temp7 = NE2temp6 + 4000\n"
  "	NE2temp8 = NE2temp7 + 4000\n"
  "	NE2temp9 = NE2temp8 + 4000\n"
  "	NE2temp10 = NE2temp9 + 4000\n"
  "	NE2temp11 = NE2temp10 + 4000\n"
  "	NE2temp12 = NE2temp11 + 4000 \n"
  "	NE2temp13 = NE2temp12 + 4000\n"
  "	NE2temp14 = NE2temp13 + 4000\n"
  "	NE2temp15 = NE2temp14 + 4000\n"
  "	\n"
  "	if (t <= NEstart2) { NE2 = 1.0}\n"
  "	else if (t >= NEstart2 && t <= NEstop2) {NE2 = NE_S }					: 1st shock\n"
  "		else if (t > NEstop2 && t < NE2temp1) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NEstop2+500)))} \n"
  "	else if (t >= NE2temp1 && t <= NE2temp1+100) {NE2=NE_S}					: 2nd shock\n"
  "		else if (t > NE2temp1+100 && t < NE2temp2) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp1+100)))}   				 \n"
  "	else if (t >= NE2temp2 && t <= NE2temp2+100) {NE2=NE_S}					: 3rd shock\n"
  "		else if (t > NE2temp2+100 && t < NE2temp3) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp2+100)))}  				 \n"
  "	else if (t >= NE2temp3 && t <= NE2temp3+100) {NE2=NE_S}					: 4th shock\n"
  "		else if (t > NE2temp3+100 && t < NE2temp4) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp3+100)))}  				 \n"
  "	else if (t >= NE2temp4 && t <= NE2temp4+100) {NE2=NE_S}					: 5th shock\n"
  "		else if (t > NE2temp4+100 && t < NE2temp5) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp4+100)))}  				 \n"
  "	else if (t >= NE2temp5 && t <= NE2temp5+100) {NE2=NE_S}					: 6th shock\n"
  "		else if (t > NE2temp5+100 && t < NE2temp6) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp5+100)))} 				 \n"
  "	else if (t >= NE2temp6 && t <= NE2temp6+100) {NE2=NE_S}					: 7th shock\n"
  "		else if (t > NE2temp6+100 && t < NE2temp7) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp6+100)))}  				 \n"
  "	else if (t >= NE2temp7 && t <= NE2temp7+100) {NE2=NE_S}					: 8th shock\n"
  "		else if (t > NE2temp7+100 && t < NE2temp8) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp7+100)))}  				    \n"
  "	else if (t >= NE2temp8 && t <= NE2temp8+100) {NE2=NE_S }					: 9th shock\n"
  "		else if (t > NE2temp8+100 && t < NE2temp9) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp8+100)))}  				    \n"
  "	else if (t >= NE2temp9 && t <= NE2temp9+100) {NE2=NE_S }					: 10th shock\n"
  "		else if (t > NE2temp9+100 && t < NE2temp10) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp9+100)))}  				    \n"
  "	else if (t >= NE2temp10 && t <= NE2temp10+100) {NE2=NE_S}					: 11th shock\n"
  "		else if (t > NE2temp10+100 && t < NE2temp11) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp10+100)))}  				 \n"
  "	else if (t >= NE2temp11 && t <= NE2temp11+100) {NE2=NE_S }					: 12th shock\n"
  "		else if (t > NE2temp11+100 && t < NE2temp12) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp11+100)))}  				 \n"
  "	else if (t >= NE2temp12 && t <= NE2temp12+100) {NE2=NE_S}					: 13th shock\n"
  "		else if (t > NE2temp12+100 && t < NE2temp13) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp12+100)))} 				 \n"
  "	else if (t >= NE2temp13 && t <= NE2temp13+100) {NE2=NE_S }					: 14th shock\n"
  "		else if (t > NE2temp13+100 && t < NE2temp14) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp13+100)))}   				 \n"
  "	else if (t >= NE2temp14 && t <= NE2temp14+100) {NE2=NE_S}					: 15th shock\n"
  "		else if (t > NE2temp14+100 && t < NE2temp15) {NE2 = 1.0 + (NE_S-1)*exp(-Beta2*(t-(NE2temp14+100)))}  				 \n"
  "	else if (t >= NE2temp15 && t <= NE2temp15+100) {NE2=NE_S}					: 16th shock\n"
  "		else  {	NE2 = 1.0}\n"
  "}\n"
  "FUNCTION GAP1(GAPstart1 (ms), GAPstop1 (ms)) {\n"
  "	LOCAL s\n"
  "	if (t <= GAPstart1) { GAP1 = 1}\n"
  "	else if (t >= GAPstop1 ) {GAP1 = 1}					: During the Gap, apply lamda2*2\n"
  "	else  {	GAP1 = 1}\n"
  "}\n"
  ":FUNCTION unirand() {    : uniform random numbers between 0 and 1\n"
  ":        unirand = scop_random()\n"
  ":}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
