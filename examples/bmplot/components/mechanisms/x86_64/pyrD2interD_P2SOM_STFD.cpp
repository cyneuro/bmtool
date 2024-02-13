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
 
#define nrn_init _nrn_init__pyrD2interD_P2SOM_STFD
#define _nrn_initial _nrn_initial__pyrD2interD_P2SOM_STFD
#define nrn_cur _nrn_cur__pyrD2interD_P2SOM_STFD
#define _nrn_current _nrn_current__pyrD2interD_P2SOM_STFD
#define nrn_jacob _nrn_jacob__pyrD2interD_P2SOM_STFD
#define nrn_state _nrn_state__pyrD2interD_P2SOM_STFD
#define _net_receive _net_receive__pyrD2interD_P2SOM_STFD 
#define release release__pyrD2interD_P2SOM_STFD 
 
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
#define P0n _p[18]
#define P0n_columnindex 18
#define fCan _p[19]
#define fCan_columnindex 19
#define P0a _p[20]
#define P0a_columnindex 20
#define fCaa _p[21]
#define fCaa_columnindex 21
#define lambda1 _p[22]
#define lambda1_columnindex 22
#define lambda2 _p[23]
#define lambda2_columnindex 23
#define threshold1 _p[24]
#define threshold1_columnindex 24
#define threshold2 _p[25]
#define threshold2_columnindex 25
#define initW _p[26]
#define initW_columnindex 26
#define fmax _p[27]
#define fmax_columnindex 27
#define fmin _p[28]
#define fmin_columnindex 28
#define thr_rp _p[29]
#define thr_rp_columnindex 29
#define facfactor _p[30]
#define facfactor_columnindex 30
#define Flimit _p[31]
#define Flimit_columnindex 31
#define f _p[32]
#define f_columnindex 32
#define tauF _p[33]
#define tauF_columnindex 33
#define d1 _p[34]
#define d1_columnindex 34
#define tauD1 _p[35]
#define tauD1_columnindex 35
#define d2 _p[36]
#define d2_columnindex 36
#define tauD2 _p[37]
#define tauD2_columnindex 37
#define ACH _p[38]
#define ACH_columnindex 38
#define bACH _p[39]
#define bACH_columnindex 39
#define inmda _p[40]
#define inmda_columnindex 40
#define g_nmda _p[41]
#define g_nmda_columnindex 41
#define on_nmda _p[42]
#define on_nmda_columnindex 42
#define W_nmda _p[43]
#define W_nmda_columnindex 43
#define iampa _p[44]
#define iampa_columnindex 44
#define g_ampa _p[45]
#define g_ampa_columnindex 45
#define on_ampa _p[46]
#define on_ampa_columnindex 46
#define limitW _p[47]
#define limitW_columnindex 47
#define ICan _p[48]
#define ICan_columnindex 48
#define ICaa _p[49]
#define ICaa_columnindex 49
#define Icatotal _p[50]
#define Icatotal_columnindex 50
#define Wmax _p[51]
#define Wmax_columnindex 51
#define Wmin _p[52]
#define Wmin_columnindex 52
#define maxChange _p[53]
#define maxChange_columnindex 53
#define normW _p[54]
#define normW_columnindex 54
#define scaleW _p[55]
#define scaleW_columnindex 55
#define pregid _p[56]
#define pregid_columnindex 56
#define postgid _p[57]
#define postgid_columnindex 57
#define F _p[58]
#define F_columnindex 58
#define D1 _p[59]
#define D1_columnindex 59
#define D2 _p[60]
#define D2_columnindex 60
#define r_nmda _p[61]
#define r_nmda_columnindex 61
#define r_ampa _p[62]
#define r_ampa_columnindex 62
#define capoolcon _p[63]
#define capoolcon_columnindex 63
#define W _p[64]
#define W_columnindex 64
#define eca _p[65]
#define eca_columnindex 65
#define t0 _p[66]
#define t0_columnindex 66
#define Afactor _p[67]
#define Afactor_columnindex 67
#define dW_ampa _p[68]
#define dW_ampa_columnindex 68
#define rp _p[69]
#define rp_columnindex 69
#define tsyn _p[70]
#define tsyn_columnindex 70
#define fa _p[71]
#define fa_columnindex 71
#define Dr_nmda _p[72]
#define Dr_nmda_columnindex 72
#define Dr_ampa _p[73]
#define Dr_ampa_columnindex 73
#define Dcapoolcon _p[74]
#define Dcapoolcon_columnindex 74
#define DW _p[75]
#define DW_columnindex 75
#define v _p[76]
#define v_columnindex 76
#define _g _p[77]
#define _g_columnindex 77
#define _tsav _p[78]
#define _tsav_columnindex 78
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
 {"eta", _hoc_eta},
 {"omega", _hoc_omega},
 {"sfunc", _hoc_sfunc},
 {"unirand", _hoc_unirand},
 {0, 0}
};
#define eta eta_pyrD2interD_P2SOM_STFD
#define omega omega_pyrD2interD_P2SOM_STFD
#define sfunc sfunc_pyrD2interD_P2SOM_STFD
#define unirand unirand_pyrD2interD_P2SOM_STFD
 extern double eta( _threadargsprotocomma_ double );
 extern double omega( _threadargsprotocomma_ double , double , double );
 extern double sfunc( _threadargsprotocomma_ double );
 extern double unirand( _threadargsproto_ );
 /* declare global and static user variables */
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
 {"ICan", "nA"},
 {"ICaa", "nA"},
 {"Icatotal", "nA"},
 {0, 0}
};
 static double W0 = 0;
 static double capoolcon0 = 0;
 static double delta_t = 0.01;
 static double r_ampa0 = 0;
 static double r_nmda0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
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
"pyrD2interD_P2SOM_STFD",
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
 "P0n",
 "fCan",
 "P0a",
 "fCaa",
 "lambda1",
 "lambda2",
 "threshold1",
 "threshold2",
 "initW",
 "fmax",
 "fmin",
 "thr_rp",
 "facfactor",
 "Flimit",
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
 "ICan",
 "ICaa",
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
 	_p = nrn_prop_data_alloc(_mechtype, 79, _prop);
 	/*initialize range parameters*/
 	srcid = -1;
 	destid = -1;
 	type = -1;
 	Cdur_nmda = 16.765;
 	AlphaTmax_nmda = 0.2659;
 	Beta_nmda = 0.008;
 	Erev_nmda = 0;
 	gbar_nmda = 0.0005;
 	Cdur_ampa = 0.713;
 	AlphaTmax_ampa = 2.257;
 	Beta_ampa = 0.0926;
 	Erev_ampa = 0;
 	gbar_ampa = 0.001;
 	Cainf = 5e-05;
 	pooldiam = 1.8172;
 	z = 2;
 	neuroM = 0;
 	tauCa = 50;
 	P0n = 0.015;
 	fCan = 0.024;
 	P0a = 0.001;
 	fCaa = 0.024;
 	lambda1 = 8;
 	lambda2 = 0.01;
 	threshold1 = 0.35;
 	threshold2 = 0.4;
 	initW = 1.5;
 	fmax = 4;
 	fmin = 0.8;
 	thr_rp = 1;
 	facfactor = 1;
 	Flimit = 10;
 	f = 1.5;
 	tauF = 150;
 	d1 = 1;
 	tauD1 = 40;
 	d2 = 1;
 	tauD2 = 70;
 	ACH = 1;
 	bACH = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 79;
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

 extern "C" void _pyrD2interD_P2SOM_STFD_reg() {
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
  hoc_register_prop_size(_mechtype, 79, 5);
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
 	ivoc_help("help ?1 pyrD2interD_P2SOM_STFD /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/pyrD2interD_P2SOM_STFD.mod\n");
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
   Dcapoolcon = - fCan * Afactor * Icatotal + ( Cainf - capoolcon ) / tauCa ;
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
    capoolcon = capoolcon + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tauCa)))*(- ( ( ( - fCan )*( Afactor ) )*( Icatotal ) + ( ( Cainf ) ) / tauCa ) / ( ( ( ( - 1.0 ) ) ) / tauCa ) - capoolcon) ;
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
     F = 1.0 + ( F - 1.0 ) * exp ( - ( t - tsyn ) / tauF ) ;
     D1 = 1.0 - ( 1.0 - D1 ) * exp ( - ( t - tsyn ) / tauD1 ) ;
     D2 = 1.0 - ( 1.0 - D2 ) * exp ( - ( t - tsyn ) / tauD2 ) ;
     tsyn = t ;
     facfactor = F * D1 * D2 ;
     F = F * f ;
     if ( F > Flimit ) {
       F = Flimit ;
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
   t0 = - 1.0 ;
   Wmax = fmax * initW ;
   Wmin = fmin * initW ;
   maxChange = ( Wmax - Wmin ) / 10.0 ;
   dW_ampa = 0.0 ;
   capoolcon = Cainf ;
   Afactor = 1.0 / ( z * FARADAY * 4.0 / 3.0 * pilocal * pow( ( pooldiam / 2.0 ) , 3.0 ) ) * ( 1e6 ) ;
   tsyn = 0.0 ;
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
   if ( neuroM  == 0.0 ) {
     g_nmda = gbar_nmda * r_nmda * facfactor ;
     }
   else {
     g_nmda = gbar_nmda * r_nmda * facfactor ;
     }
   inmda = W_nmda * g_nmda * ( v - Erev_nmda ) * sfunc ( _threadargscomma_ v ) ;
   g_ampa = gbar_ampa * r_ampa * facfactor ;
   iampa = W * g_ampa * ( v - Erev_ampa ) * ( 1.0 + ( bACH * ( ACH - 1.0 ) ) ) ;
   ICan = P0n * g_nmda * ( v - eca ) * sfunc ( _threadargscomma_ v ) ;
   ICaa = P0a * W * g_ampa * ( v - eca ) / initW ;
   Icatotal = ICan + ICaa ;
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
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/pyrD2interD_P2SOM_STFD.mod";
    const char* nmodl_file_text = 
  ":Pyramidal Cells to Interneuron Cells AMPA+NMDA with local Ca2+ pool\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS pyrD2interD_P2SOM_STFD\n"
  "	USEION ca READ eca	\n"
  "	NONSPECIFIC_CURRENT inmda, iampa\n"
  "	RANGE initW\n"
  "	RANGE Cdur_nmda, AlphaTmax_nmda, Beta_nmda, Erev_nmda, gbar_nmda, W_nmda, on_nmda, g_nmda\n"
  "	RANGE Cdur_ampa, AlphaTmax_ampa, Beta_ampa, Erev_ampa, gbar_ampa, W, on_ampa, g_ampa\n"
  "	RANGE eca, ICan, P0n, fCan, tauCa, Icatotal\n"
  "	RANGE ICaa, P0a, fCaa\n"
  "	RANGE Cainf, pooldiam, z\n"
  "	RANGE lambda1, lambda2, threshold1, threshold2\n"
  "	RANGE fmax, fmin, Wmax, Wmin, maxChange, normW, scaleW, srcid, destid,limitW\n"
  "	RANGE pregid,postgid, thr_rp\n"
  "	RANGE F, f, tauF, D1, d1, tauD1, D2, d2, tauD2,Flimit\n"
  "	RANGE facfactor\n"
  "	RANGE neuroM,type\n"
  "	RANGE ACH, bACH\n"
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
  "	Cdur_nmda = 16.7650 (ms)\n"
  "	AlphaTmax_nmda = .2659 (/ms)\n"
  "	Beta_nmda = 0.008 (/ms)\n"
  "	Erev_nmda = 0 (mV)\n"
  "	gbar_nmda = .5e-3 (uS)\n"
  "\n"
  "	Cdur_ampa = 0.713 (ms)\n"
  "	AlphaTmax_ampa = 2.257(/ms) :1.1286 10.1571\n"
  "	Beta_ampa = 0.0926 (/ms) :0.0463 0.4167\n"
  "	Erev_ampa = 0 (mV)\n"
  "	gbar_ampa = 1e-3 (uS)\n"
  "\n"
  "	eca = 120\n"
  "\n"
  "	Cainf = 50e-6 (mM)\n"
  "	pooldiam =  1.8172 (micrometer)\n"
  "	z = 2\n"
  "	neuroM = 0\n"
  "\n"
  "	tauCa = 50 (ms)\n"
  "	P0n = .015\n"
  "	fCan = .024\n"
  "	\n"
  "	P0a = .001\n"
  "	fCaa = .024\n"
  "	\n"
  "	lambda1 = 8 : 3 : 10 :6 : 4 :2\n"
  "	lambda2 = .01\n"
  "	threshold1 = 0.35 : 0.4 :  0.45 :0.5 (uM)\n"
  "	threshold2 = 0.4 : 0.45 :  0.5 :0.6 (uM)\n"
  "\n"
  "	:AMPA Weight\n"
  "	initW = 1.5 : 1.5 : 2 : 0.1:3 : 2 :3\n"
  "	fmax = 4 : 8 : 5: 4 :3\n"
  "	fmin = .8	\n"
  "\n"
  "	thr_rp = 1 : .7\n"
  "	\n"
  "	facfactor = 1\n"
  "    Flimit=10\n"
  "	: the (1) is needed for the range limits to be effective\n"
  "        f = 1.5 (1) < 0, 1e9 >    : facilitation  : 1.3 (1) < 0, 1e9 >    : facilitation\n"
  "        tauF = 150 (ms) < 1e-9, 1e9 >\n"
  "        d1 = 1 (1) < 0, 1 >: 0.95 (1) < 0, 1 >     : fast depression\n"
  "        tauD1 = 40 (ms) < 1e-9, 1e9 >\n"
  "        d2 = 1 (1) < 0, 1 > : 0.9 (1) < 0, 1 >     : slow depression\n"
  "        tauD2 = 70 (ms) < 1e-9, 1e9 >		\n"
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
  "	ICan (nA)\n"
  "	ICaa (nA)\n"
  "	Afactor	(mM/ms/nA)\n"
  "	Icatotal (nA)\n"
  "\n"
  "	dW_ampa\n"
  "	Wmax\n"
  "	Wmin\n"
  "	maxChange\n"
  "	normW\n"
  "	scaleW\n"
  "	\n"
  "	pregid\n"
  "	postgid\n"
  "	\n"
  "	rp\n"
  "	tsyn\n"
  "	\n"
  "	fa\n"
  "	F\n"
  "	D1\n"
  "	D2		\n"
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
  "	limitW = 1\n"
  "\n"
  "	t0 = -1\n"
  "\n"
  "	Wmax = fmax*initW\n"
  "	Wmin = fmin*initW\n"
  "	maxChange = (Wmax-Wmin)/10\n"
  "	dW_ampa = 0\n"
  "\n"
  "	capoolcon = Cainf\n"
  "	Afactor	= 1/(z*FARADAY*4/3*pilocal*(pooldiam/2)^3)*(1e6)\n"
  "	\n"
  "    tsyn=0    \n"
  "	fa =0\n"
  "	F = 1\n"
  "	D1 = 1\n"
  "	D2 = 1		\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  " if ((eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W))>0&&W>=Wmax) {\n"
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
  "     : if (W >= Wmax || W <= Wmin ) {     : for limiting the weight\n"
  "	 : limitW=1e-12\n"
  "	 : } else {\n"
  "	  : limitW=1\n"
  "	 : }\n"
  "	: if (W > Wmax) { \n"
  "	:	W = Wmax\n"
  "	: } else if (W < Wmin) {\n"
  " 		: W = Wmin\n"
  "	: }\n"
  "	\n"
  "	 if (neuroM==0) {\n"
  "	g_nmda = gbar_nmda*r_nmda*facfactor\n"
  "	} else {\n"
  "	g_nmda = gbar_nmda*r_nmda*facfactor\n"
  "	}\n"
  "	inmda = W_nmda*g_nmda*(v - Erev_nmda)*sfunc(v)\n"
  "\n"
  "	g_ampa = gbar_ampa*r_ampa*facfactor\n"
  "	iampa = W*g_ampa*(v - Erev_ampa)*(1 + (bACH * (ACH - 1)))\n"
  "\n"
  "	ICan = P0n*g_nmda*(v - eca)*sfunc(v)\n"
  "	ICaa = P0a*W*g_ampa*(v-eca)/initW	\n"
  "	Icatotal = ICan + ICaa\n"
  "}\n"
  "\n"
  "DERIVATIVE release {\n"
  "    : W' = eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)\n"
  "   \n"
  "	    W' = 1e-12*limitW*eta(capoolcon)*(lambda1*omega(capoolcon, threshold1, threshold2)-lambda2*W)	  : Long-term plasticity was implemented. (Shouval et al. 2002a, 2002b)\n"
  "\n"
  "	r_nmda' = AlphaTmax_nmda*on_nmda*(1-r_nmda)-Beta_nmda*r_nmda\n"
  "	r_ampa' = AlphaTmax_ampa*on_ampa*(1-r_ampa)-Beta_ampa*r_ampa\n"
  "  \n"
  "	capoolcon'= -fCan*Afactor*Icatotal + (Cainf-capoolcon)/tauCa\n"
  "}\n"
  "\n"
  "NET_RECEIVE(dummy_weight) {\n"
  "\n"
  " if (flag==0) {           :a spike arrived, start onset state if not already on\n"
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
  "	\n"
  "	if (flag == 0) {  : Short term plasticity was implemented(Varela et. al 1997):\n"
  "	\n"
  "	rp = unirand()	\n"
  "	\n"
  "	F  = 1 + (F-1)* exp(-(t - tsyn)/tauF)\n"
  "	D1 = 1 - (1-D1)*exp(-(t - tsyn)/tauD1)\n"
  "	D2 = 1 - (1-D2)*exp(-(t - tsyn)/tauD2)\n"
  " :printf(\"%g\\t%g\\t%g\\t%g\\t%g\\t%g\\n\", t, t-tsyn, F, D1, D2, facfactor)\n"
  "	::printf(\"%g\\t%g\\t%g\\t%g\\n\", F, D1, D2, facfactor)\n"
  "	tsyn = t\n"
  "	\n"
  "	facfactor = F * D1 * D2\n"
  "\n"
  "	F = F * f\n"
  "	\n"
  "	if (F > Flimit) { \n"
  "	F=Flimit\n"
  "	}\n"
  "	:if (facfactor < 0.7) { \n"
  "	:facfactor=0.7\n"
  "	:}\n"
  "	:if (F < 0.8) { \n"
  "	:F=0.8\n"
  "	:}	\n"
  "	D1 = D1 * d1\n"
  "	D2 = D2 * d2\n"
  ":printf(\"\\t%g\\t%g\\t%g\\n\", F, D1, D2)\n"
  "}\n"
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
  "FUNCTION unirand() {    : uniform random numbers between 0 and 1\n"
  "        unirand = 0: can be set to 0 since thr_rp is set to 1 anyway\n"
  "}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
