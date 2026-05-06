/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__kBK
#define _nrn_initial _nrn_initial__kBK
#define nrn_cur _nrn_cur__kBK
#define _nrn_current _nrn_current__kBK
#define nrn_jacob _nrn_jacob__kBK
#define nrn_state _nrn_state__kBK
#define _net_receive _net_receive__kBK 
#define rate rate__kBK 
#define states states__kBK 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gpeak _p[0]
#define gpeak_columnindex 0
#define caPh _p[1]
#define caPh_columnindex 1
#define caPk _p[2]
#define caPk_columnindex 2
#define caPmax _p[3]
#define caPmax_columnindex 3
#define caPmin _p[4]
#define caPmin_columnindex 4
#define caVhh _p[5]
#define caVhh_columnindex 5
#define caVhmax _p[6]
#define caVhmax_columnindex 6
#define caVhmin _p[7]
#define caVhmin_columnindex 7
#define k _p[8]
#define k_columnindex 8
#define tau _p[9]
#define tau_columnindex 9
#define p _p[10]
#define p_columnindex 10
#define ek _p[11]
#define ek_columnindex 11
#define ik _p[12]
#define ik_columnindex 12
#define cai _p[13]
#define cai_columnindex 13
#define caiScaled _p[14]
#define caiScaled_columnindex 14
#define pinf _p[15]
#define pinf_columnindex 15
#define Dp _p[16]
#define Dp_columnindex 16
#define v _p[17]
#define v_columnindex 17
#define _g _p[18]
#define _g_columnindex 18
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
#define _ion_cai	*_ppvar[3]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_P0ca(void);
 static void _hoc_Vhca(void);
 static void _hoc_rate(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_kBK", _hoc_setdata,
 "P0ca_kBK", _hoc_P0ca,
 "Vhca_kBK", _hoc_Vhca,
 "rate_kBK", _hoc_rate,
 0, 0
};
#define P0ca P0ca_kBK
#define Vhca Vhca_kBK
 extern double P0ca( _threadargsprotocomma_ double );
 extern double Vhca( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define caVhk caVhk_kBK
 double caVhk = -0.94208;
#define pinfmin pinfmin_kBK
 double pinfmin = 0;
#define scale scale_kBK
 double scale = 100;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gpeak_kBK", 0, 1e+09,
 "tau_kBK", 1e-12, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gpeak_kBK", "mho/cm2",
 "caPh_kBK", "mM",
 "caVhh_kBK", "mM",
 "caVhmax_kBK", "mV",
 "caVhmin_kBK", "mV",
 "k_kBK", "mV",
 "tau_kBK", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double p0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "caVhk_kBK", &caVhk_kBK,
 "scale_kBK", &scale_kBK,
 "pinfmin_kBK", &pinfmin_kBK,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"kBK",
 "gpeak_kBK",
 "caPh_kBK",
 "caPk_kBK",
 "caPmax_kBK",
 "caPmin_kBK",
 "caVhh_kBK",
 "caVhmax_kBK",
 "caVhmin_kBK",
 "k_kBK",
 "tau_kBK",
 0,
 0,
 "p_kBK",
 0,
 0};
 static Symbol* _k_sym;
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 19, _prop);
 	/*initialize range parameters*/
 	gpeak = 0.0268;
 	caPh = 0.002;
 	caPk = 1;
 	caPmax = 1;
 	caPmin = 0;
 	caVhh = 0.002;
 	caVhmax = 155.67;
 	caVhmin = -46.08;
 	k = 17;
 	tau = 1;
 	_prop->param = _p;
 	_prop->param_size = 19;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[3]._pval = &prop_ion->param[1]; /* cai */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _kBK_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	ion_reg("ca", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 19, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 kBK /Users/gregglickert/Documents/GitHub/bmtool/docs/examples/notebooks/synapses/synaptic_tuner/modfiles/kBK.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "large-conductance calcium-activated potassium channel (BK)";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rate(_threadargsprotocomma_ double, double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rate ( _threadargscomma_ v , cai ) ;
   Dp = ( pinf - p ) / tau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rate ( _threadargscomma_ v , cai ) ;
 Dp = Dp  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rate ( _threadargscomma_ v , cai ) ;
    p = p + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau)))*(- ( ( ( pinf ) ) / tau ) / ( ( ( ( - 1.0 ) ) ) / tau ) - p) ;
   }
  return 0;
}
 
static int  rate ( _threadargsprotocomma_ double _lv , double _lca ) {
   caiScaled = _lca * scale ;
   pinf = P0ca ( _threadargscomma_ caiScaled ) / ( 1.0 + exp ( ( Vhca ( _threadargscomma_ caiScaled ) - _lv ) / k ) ) ;
   if ( pinf < pinfmin ) {
     pinf = 0.0 ;
     }
    return 0; }
 
static void _hoc_rate(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rate ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double P0ca ( _threadargsprotocomma_ double _lca ) {
   double _lP0ca;
 if ( _lca < 1E-18 ) {
     _lP0ca = caPmin ;
     }
   else {
     _lP0ca = caPmin + ( ( caPmax - caPmin ) / ( 1.0 + pow( ( caPh / _lca ) , caPk ) ) ) ;
     }
   
return _lP0ca;
 }
 
static void _hoc_P0ca(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  P0ca ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double Vhca ( _threadargsprotocomma_ double _lca ) {
   double _lVhca;
 if ( _lca < 1E-18 ) {
     _lVhca = caVhmax ;
     }
   else {
     _lVhca = caVhmin + ( ( caVhmax - caVhmin ) / ( 1.0 + ( pow( ( caVhh / _lca ) , caVhk ) ) ) ) ;
     }
   
return _lVhca;
 }
 
static void _hoc_Vhca(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  Vhca ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
  cai = _ion_cai;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
  cai = _ion_cai;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  p = p0;
 {
   rate ( _threadargscomma_ v , cai ) ;
   p = pinf ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
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
 v = _v;
  ek = _ion_ek;
  cai = _ion_cai;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ik = gpeak * p * ( v - ek ) ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
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
  ek = _ion_ek;
  cai = _ion_cai;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
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

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
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

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
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
  ek = _ion_ek;
  cai = _ion_cai;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = p_columnindex;  _dlist1[0] = Dp_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/gregglickert/Documents/GitHub/bmtool/docs/examples/notebooks/synapses/synaptic_tuner/modfiles/kBK.mod";
static const char* nmodl_file_text = 
  ": from https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=168148&file=/stadler2014_layerV/kBK.mod\n"
  "TITLE large-conductance calcium-activated potassium channel (BK)\n"
  "	:Mechanism according to Gong et al 2001 and Womack&Khodakakhah 2002,\n"
  "	:adapted for Layer V cells on the basis of Benhassine&Berger 2005.\n"
  "	:NB: concentrations in mM\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX kBK\n"
  "	USEION k READ ek WRITE ik\n"
  "	USEION ca READ cai\n"
  "	RANGE gpeak, gkact, caPh, caPk, caPmax, caPmin\n"
  "	RANGE caVhh, CaVhk, caVhmax, caVhmin, k, tau\n"
  "        GLOBAL pinfmin : cutoff - if pinf < pinfmin, set to 0.; by default cutoff not used (pinfmin==0)\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(molar) = (1/liter)\n"
  "	(mM) 	= (millimolar)\n"
  "}\n"
  "\n"
  "\n"
  "PARAMETER {\n"
  "		:maximum conductance (Benhassine 05)\n"
  "	gpeak   = 268e-4	(mho/cm2) <0, 1e9>\n"
  "\n"
  "	                                    : Calcium dependence of opening probability (Gong 2001)\n"
  "	caPh    = 2e-3     (mM)             : conc. with half maximum open probaility\n"
  "	caPk    = 1                         : Steepness of calcium dependence curve\n"
  "	caPmax  = 1                         : max and\n"
  "	caPmin  = 0                         : min open probability\n"
  "\n"
  "	                                    : Calcium dependence of Vh shift (Womack 2002)\n"
  "	caVhh   = 2e-3    (mM)              : Conc. for half of the Vh shift\n"
  "	caVhk   = -0.94208                  : Steepness of the Vh-calcium dependence curve\n"
  "	caVhmax = 155.67 (mV)               : max and\n"
  "	caVhmin = -46.08 (mV)               : min Vh\n"
  "\n"
  "	                                    : Voltage dependence of open probability (Gong 2001)\n"
  "	                                    : must not be zero\n"
  "	k       = 17	(mV)\n"
  "\n"
  "	                                    : Timeconstant of channel kinetics\n"
  "	                                    : no data for a description of a calcium&voltage dependence\n"
  "	                                    : some points (room temp) in Behassine 05 & Womack 02\n"
  "	tau     = 1 (ms) <1e-12, 1e9>\n"
  "	scale   = 100                       : scaling to incorporate higher ca conc near ca channels\n"
  "\n"
  "        pinfmin = 0.0                       : cutoff for pinf - less than that set pinf to 0.0\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v 		(mV)\n"
  "	ek		(mV)\n"
  "	ik 		(mA/cm2)\n"
  "    	cai  		(mM)\n"
  "	caiScaled	(mM)\n"
  "	pinf		(1)\n"
  "}\n"
  "\n"
  "\n"
  "STATE {\n"
  "        p\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	ik = gpeak*p* (v - ek)\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "        rate(v, cai)\n"
  "        p' =  (pinf - p)/tau\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "        rate(v, cai)\n"
  "        p = pinf\n"
  "}\n"
  "\n"
  "PROCEDURE rate(v(mV), ca(mM))  {\n"
  "        caiScaled = ca*scale\n"
  "        pinf = P0ca(caiScaled) / ( 1 + exp( (Vhca(caiScaled)-v)/k ) )\n"
  "        if(pinf < pinfmin) { pinf = 0.0 }\n"
  "}\n"
  "\n"
  "FUNCTION P0ca(ca(mM)) (1) {\n"
  "\n"
  "	if (ca < 1E-18) { 		:check for division by zero\n"
  "	P0ca = caPmin\n"
  "	} else {\n"
  "	P0ca = caPmin + ( (caPmax - caPmin) / ( 1 + (caPh/ca)^caPk ))\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION Vhca(ca(mM)) (mV) {\n"
  "\n"
  "	if (ca < 1E-18) {		:check for division by zero\n"
  "	Vhca = caVhmax\n"
  "	} else {\n"
  "	Vhca = caVhmin + ( (caVhmax - caVhmin ) / ( 1 + ((caVhh/ca)^caVhk)) )\n"
  "	}\n"
  "}\n"
  ;
#endif
