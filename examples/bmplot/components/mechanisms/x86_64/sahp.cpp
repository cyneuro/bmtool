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
 
#define nrn_init _nrn_init__sAHP
#define _nrn_initial _nrn_initial__sAHP
#define nrn_cur _nrn_cur__sAHP
#define _nrn_current _nrn_current__sAHP
#define nrn_jacob _nrn_jacob__sAHP
#define nrn_state _nrn_state__sAHP
#define _net_receive _net_receive__sAHP 
#define rate rate__sAHP 
#define states states__sAHP 
 
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
#define gsAHPbar _p[0]
#define gsAHPbar_columnindex 0
#define i _p[1]
#define i_columnindex 1
#define cinf _p[2]
#define cinf_columnindex 2
#define ctau _p[3]
#define ctau_columnindex 3
#define gk _p[4]
#define gk_columnindex 4
#define c _p[5]
#define c_columnindex 5
#define ek _p[6]
#define ek_columnindex 6
#define casi _p[7]
#define casi_columnindex 7
#define ik _p[8]
#define ik_columnindex 8
#define Dc _p[9]
#define Dc_columnindex 9
#define v _p[10]
#define v_columnindex 10
#define _g _p[11]
#define _g_columnindex 11
#define _ion_ek	*(_ppvar[0].get<double*>())
#define _ion_ik	*_ppvar[1].get<double*>()
#define _ion_dikdv	*_ppvar[2].get<double*>()
#define _ion_casi	*(_ppvar[3].get<double*>())
 
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
 static void _hoc_cbet(void);
 static void _hoc_calf(void);
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
static void register_nmodl_text_and_filename(int mechtype);
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
 {"setdata_sAHP", _hoc_setdata},
 {"cbet_sAHP", _hoc_cbet},
 {"calf_sAHP", _hoc_calf},
 {"rate_sAHP", _hoc_rate},
 {0, 0}
};
#define cbet cbet_sAHP
#define calf calf_sAHP
 extern double cbet( _threadargsprotocomma_ double , double );
 extern double calf( _threadargsprotocomma_ double , double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"i_sAHP", "mA/cm2"},
 {"ctau_sAHP", "ms"},
 {"gk_sAHP", "mho/cm2"},
 {0, 0}
};
 static double c0 = 0;
 static double delta_t = 0.01;
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
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, Memb_list*, int);
static void _ode_matsol(NrnThread*, Memb_list*, int);
 
#define _cvode_ieq _ppvar[4].literal_value<int>()
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"sAHP",
 "gsAHPbar_sAHP",
 0,
 "i_sAHP",
 "cinf_sAHP",
 "ctau_sAHP",
 "gk_sAHP",
 0,
 "c_sAHP",
 0,
 0};
 static Symbol* _k_sym;
 static Symbol* _cas_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 12, _prop);
 	/*initialize range parameters*/
 	gsAHPbar = 2.31814e-05;
 	_prop->param = _p;
 	_prop->param_size = 12;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0] = &prop_ion->param[0]; /* ek */
 	_ppvar[1] = &prop_ion->param[3]; /* ik */
 	_ppvar[2] = &prop_ion->param[4]; /* _ion_dikdv */
 prop_ion = need_memb(_cas_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[3] = &prop_ion->param[1]; /* casi */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _sahp_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	ion_reg("cas", 2.0);
 	_k_sym = hoc_lookup("k_ion");
 	_cas_sym = hoc_lookup("cas_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
  hoc_register_prop_size(_mechtype, 12, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cas_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 sAHP /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/sahp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "";

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
   rate ( _threadargscomma_ v , casi ) ;
   Dc = ( cinf - c ) / ctau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rate ( _threadargscomma_ v , casi ) ;
 Dc = Dc  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ctau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rate ( _threadargscomma_ v , casi ) ;
    c = c + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ctau)))*(- ( ( ( cinf ) ) / ctau ) / ( ( ( ( - 1.0 ) ) ) / ctau ) - c) ;
   }
  return 0;
}
 
double calf ( _threadargsprotocomma_ double _lv , double _lcasi ) {
   double _lcalf;
 double _lvs , _lva ;
  _lvs = 10.0 * log10 ( 1000.0 * _lcasi ) ;
   _lcalf = 0.0048 / exp ( - 0.5 * ( _lvs - 35.0 ) ) ;
    
return _lcalf;
 }
 
static void _hoc_calf(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  calf ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double cbet ( _threadargsprotocomma_ double _lv , double _lcasi ) {
   double _lcbet;
 double _lvs , _lvb ;
  _lvs = 10.0 * log10 ( 1000.0 * _lcasi ) ;
   _lcbet = 0.012 / exp ( 0.2 * ( _lvs + 100.0 ) ) ;
    
return _lcbet;
 }
 
static void _hoc_cbet(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  cbet ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int  rate ( _threadargsprotocomma_ double _lv , double _lcasi ) {
   double _lcsum , _lca , _lcb ;
  _lca = calf ( _threadargscomma_ _lv , _lcasi ) ;
   _lcb = cbet ( _threadargscomma_ _lv , _lcasi ) ;
   _lcsum = _lca + _lcb ;
   if ( _lv < - 65.0 ) {
     cinf = 0.0 ;
     }
   else {
     cinf = _lca / _lcsum ;
     }
   ctau = 48.0 ;
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
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(NrnThread* _nt, Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
  casi = _ion_casi;
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
 
static void _ode_matsol(NrnThread* _nt, Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
  casi = _ion_casi;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
   nrn_update_ion_pointer(_cas_sym, _ppvar, 3, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  c = c0;
 {
   rate ( _threadargscomma_ v , casi ) ;
   c = cinf ;
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
  casi = _ion_casi;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gk = gsAHPbar * c ;
   ik = gk * ( v - ek ) ;
   i = ik ;
   }
 _current += ik;

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
  ek = _ion_ek;
  casi = _ion_casi;
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
  ek = _ion_ek;
  casi = _ion_casi;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = c_columnindex;  _dlist1[0] = Dc_columnindex;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/sahp.mod";
    const char* nmodl_file_text = 
  ":  iC   fast Ca2+/V-dependent K+ channel\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX sAHP\n"
  "	USEION k READ ek WRITE ik\n"
  "	USEION cas READ casi VALENCE 2 \n"
  "        RANGE gk, i , ctau, cinf, gsAHPbar : ,ik\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "        (mM) = (milli/liter)\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	gsAHPbar= 2.318144e-05 : 0.0001	(mho/cm2) : \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)\n"
  "	ek (mV)\n"
  "	casi (mM)\n"
  "	ik (mA/cm2)\n"
  "	i  (mA/cm2)\n"
  "	cinf \n"
  "	ctau (ms)\n"
  "	gk (mho/cm2)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	c\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	gk = gsAHPbar*c       \n"
  "	ik = gk*(v-ek)\n"
  "	i = ik\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rate(v,casi)\n"
  "	c = cinf\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "        rate(v,casi)\n"
  "	c' = (cinf-c)/ctau\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "\n"
  "FUNCTION calf(v (mV), casi (mM)) (/ms) { LOCAL vs, va\n"
  "	UNITSOFF\n"
  "	vs=10*log10(1000*casi)\n"
  "	calf = 0.0048/exp(-0.5*(vs-35))\n"
  "	UNITSON\n"
  "}\n"
  "\n"
  "FUNCTION cbet(v (mV), casi (mM))(/ms) { LOCAL vs, vb \n"
  "	UNITSOFF\n"
  "	  vs=10*log10(1000*casi)\n"
  "	  cbet = 0.012/exp(0.2*(vs+100))\n"
  "	UNITSON\n"
  "}\n"
  "\n"
  "UNITSON\n"
  "\n"
  "PROCEDURE rate(v (mV), casi (mM)) {LOCAL  csum, ca, cb\n"
  "	UNITSOFF\n"
  "	ca=calf(v, casi) \n"
  "	cb=cbet(v, casi)		\n"
  "	csum = ca+cb\n"
  "	if (v < -65 ) {              :::::::::::::::::::::::::::  67.5\n"
  "	cinf = 0\n"
  "	} else{\n"
  "	cinf = ca/csum\n"
  "	}\n"
  "	ctau = 48\n"
  "	UNITSON\n"
  "}	\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
