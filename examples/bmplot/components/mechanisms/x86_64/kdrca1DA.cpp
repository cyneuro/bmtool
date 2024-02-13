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
 
#define nrn_init _nrn_init__kdrDA
#define _nrn_initial _nrn_initial__kdrDA
#define nrn_cur _nrn_cur__kdrDA
#define _nrn_current _nrn_current__kdrDA
#define nrn_jacob _nrn_jacob__kdrDA
#define nrn_state _nrn_state__kdrDA
#define _net_receive _net_receive__kdrDA 
#define rates rates__kdrDA 
#define states states__kdrDA 
 
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
#define gbar _p[0]
#define gbar_columnindex 0
#define i _p[1]
#define i_columnindex 1
#define ninf _p[2]
#define ninf_columnindex 2
#define gkdr _p[3]
#define gkdr_columnindex 3
#define taun _p[4]
#define taun_columnindex 4
#define n _p[5]
#define n_columnindex 5
#define ek _p[6]
#define ek_columnindex 6
#define Dn _p[7]
#define Dn_columnindex 7
#define ik _p[8]
#define ik_columnindex 8
#define v _p[9]
#define v_columnindex 9
#define _g _p[10]
#define _g_columnindex 10
#define _ion_ek	*(_ppvar[0].get<double*>())
#define _ion_ik	*_ppvar[1].get<double*>()
#define _ion_dikdv	*_ppvar[2].get<double*>()
 
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
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_DA2(void);
 static void _hoc_DA1(void);
 static void _hoc_alpn(void);
 static void _hoc_betn(void);
 static void _hoc_rates(void);
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
 {"setdata_kdrDA", _hoc_setdata},
 {"DA2_kdrDA", _hoc_DA2},
 {"DA1_kdrDA", _hoc_DA1},
 {"alpn_kdrDA", _hoc_alpn},
 {"betn_kdrDA", _hoc_betn},
 {"rates_kdrDA", _hoc_rates},
 {0, 0}
};
#define DA2 DA2_kdrDA
#define DA1 DA1_kdrDA
#define alpn alpn_kdrDA
#define betn betn_kdrDA
 extern double DA2( _threadargsprotocomma_ double );
 extern double DA1( _threadargsprotocomma_ double );
 extern double alpn( _threadargsprotocomma_ double );
 extern double betn( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define DA_t2 DA_t2_kdrDA
 double DA_t2 = 0.8;
#define DA_start2 DA_start2_kdrDA
 double DA_start2 = 36000;
#define DA_period2 DA_period2_kdrDA
 double DA_period2 = 100;
#define DA_t1 DA_t1_kdrDA
 double DA_t1 = 0.95;
#define DA_ext2 DA_ext2_kdrDA
 double DA_ext2 = 212000;
#define DA_ext1 DA_ext1_kdrDA
 double DA_ext1 = 196000;
#define DA_stop DA_stop_kdrDA
 double DA_stop = 96000;
#define DA_start DA_start_kdrDA
 double DA_start = 64000;
#define DA_period DA_period_kdrDA
 double DA_period = 500;
#define a0n a0n_kdrDA
 double a0n = 0.02;
#define gmn gmn_kdrDA
 double gmn = 0.7;
#define nmax nmax_kdrDA
 double nmax = 2;
#define qt qt_kdrDA
 double qt = 1;
#define tone_period tone_period_kdrDA
 double tone_period = 4000;
#define vhalfn vhalfn_kdrDA
 double vhalfn = -15;
#define zetan zetan_kdrDA
 double zetan = -3;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"a0n_kdrDA", "/ms"},
 {"zetan_kdrDA", "1"},
 {"gmn_kdrDA", "1"},
 {"nmax_kdrDA", "1"},
 {"gbar_kdrDA", "mho/cm2"},
 {"i_kdrDA", "mA/cm2"},
 {0, 0}
};
 static double delta_t = 0.01;
 static double n0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"tone_period_kdrDA", &tone_period_kdrDA},
 {"DA_period_kdrDA", &DA_period_kdrDA},
 {"DA_start_kdrDA", &DA_start_kdrDA},
 {"DA_stop_kdrDA", &DA_stop_kdrDA},
 {"DA_ext1_kdrDA", &DA_ext1_kdrDA},
 {"DA_ext2_kdrDA", &DA_ext2_kdrDA},
 {"DA_t1_kdrDA", &DA_t1_kdrDA},
 {"DA_period2_kdrDA", &DA_period2_kdrDA},
 {"DA_start2_kdrDA", &DA_start2_kdrDA},
 {"DA_t2_kdrDA", &DA_t2_kdrDA},
 {"vhalfn_kdrDA", &vhalfn_kdrDA},
 {"a0n_kdrDA", &a0n_kdrDA},
 {"zetan_kdrDA", &zetan_kdrDA},
 {"gmn_kdrDA", &gmn_kdrDA},
 {"nmax_kdrDA", &nmax_kdrDA},
 {"qt_kdrDA", &qt_kdrDA},
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
 
#define _cvode_ieq _ppvar[3].literal_value<int>()
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"kdrDA",
 "gbar_kdrDA",
 0,
 "i_kdrDA",
 "ninf_kdrDA",
 "gkdr_kdrDA",
 "taun_kdrDA",
 0,
 "n_kdrDA",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 11, _prop);
 	/*initialize range parameters*/
 	gbar = 0.003;
 	_prop->param = _p;
 	_prop->param_size = 11;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0] = &prop_ion->param[0]; /* ek */
 	_ppvar[1] = &prop_ion->param[3]; /* ik */
 	_ppvar[2] = &prop_ion->param[4]; /* _ion_dikdv */
 
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

 extern "C" void _kdrca1DA_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
  hoc_register_prop_size(_mechtype, 11, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 kdrDA /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/kdrca1DA.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "K-DR channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
double alpn ( _threadargsprotocomma_ double _lv ) {
   double _lalpn;
 _lalpn = exp ( 1.e-3 * ( - 3.0 ) * ( _lv - vhalfn ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lalpn;
 }
 
static void _hoc_alpn(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alpn ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double betn ( _threadargsprotocomma_ double _lv ) {
   double _lbetn;
 _lbetn = exp ( 1.e-3 * ( - 3.0 ) * ( 0.7 ) * ( _lv - vhalfn ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lbetn;
 }
 
static void _hoc_betn(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betn ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dn = ( ninf - n ) / taun ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taun )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taun)))*(- ( ( ( ninf ) ) / taun ) / ( ( ( ( - 1.0 ) ) ) / taun ) - n) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   double _la ;
 _la = alpn ( _threadargscomma_ _lv ) ;
   if ( _lv < - 55.0 ) {
     ninf = 0.0 ;
     }
   else {
     ninf = 1.0 / ( 1.0 + exp ( ( vhalfn - _lv ) / 11.0 ) ) ;
     }
   taun = betn ( _threadargscomma_ _lv ) / ( qt * ( 0.02 ) * ( 1.0 + _la ) ) ;
   if ( taun < nmax ) {
     taun = nmax ;
     }
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double DA1 ( _threadargsprotocomma_ double _lt ) {
   double _lDA1;
 if ( _lt >= DA_start  && _lt <= DA_stop ) {
     if ( ( _lt / tone_period - floor ( _lt / tone_period ) ) >= ( 1.0 - DA_period / tone_period ) ) {
       _lDA1 = DA_t1 ;
       }
     else if ( ( _lt / tone_period - floor ( _lt / tone_period ) )  == 0.0 ) {
       _lDA1 = DA_t1 ;
       }
     else {
       _lDA1 = 1.0 ;
       }
     }
   else if ( _lt >= DA_ext1  && _lt <= DA_ext2 ) {
     if ( ( _lt / tone_period - floor ( _lt / tone_period ) ) >= ( 1.0 - DA_period / tone_period ) ) {
       _lDA1 = DA_t1 ;
       }
     else if ( ( _lt / tone_period - floor ( _lt / tone_period ) )  == 0.0 ) {
       _lDA1 = DA_t1 ;
       }
     else {
       _lDA1 = 1.0 ;
       }
     }
   else {
     _lDA1 = 1.0 ;
     }
   
return _lDA1;
 }
 
static void _hoc_DA1(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  DA1 ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double DA2 ( _threadargsprotocomma_ double _lt ) {
   double _lDA2;
 if ( _lt >= DA_start2  && _lt <= DA_stop ) {
     if ( ( _lt / tone_period - floor ( _lt / tone_period ) ) >= ( 1.0 - DA_period2 / tone_period ) ) {
       _lDA2 = DA_t2 ;
       }
     else if ( ( _lt / tone_period - floor ( _lt / tone_period ) )  == 0.0 ) {
       _lDA2 = DA_t2 ;
       }
     else {
       _lDA2 = 1.0 ;
       }
     }
   else {
     _lDA2 = 1.0 ;
     }
   
return _lDA2;
 }
 
static void _hoc_DA2(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  DA2 ( _p, _ppvar, _thread, _nt, *getarg(1) );
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
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  n = n0;
 {
   rates ( _threadargscomma_ v ) ;
   n = ninf ;
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
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gkdr = gbar * n ;
   ik = gkdr * ( v - ek ) * DA1 ( _threadargscomma_ t ) * DA2 ( _threadargscomma_ t ) ;
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
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = n_columnindex;  _dlist1[0] = Dn_columnindex;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/kdrca1DA.mod";
    const char* nmodl_file_text = 
  "TITLE K-DR channel\n"
  ": from Klee Ficker and Heinemann\n"
  ": modified to account for Dax et al.\n"
  ": M.Migliore 1997\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	tone_period = 4000    \n"
  "	DA_period = 500	\n"
  "	DA_start = 64000		             : D1R(Low Affinity) Dopamine Effect after 6 conditioning trials (15*4000) = 60000)\n"
  "	DA_stop = 96000\n"
  "	DA_ext1 = 196000\n"
  "	DA_ext2 = 212000	\n"
  "	\n"
  "	DA_t1 = 0.95 : 0.9 : 1 :  1 : 0.9           : Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP\n"
  "	DA_period2 = 100\n"
  "	DA_start2 = 36000		   			: shock Dopamine Effect during shock after 1 conditioning trial\n"
  "	DA_t2 = .8           				: Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP	\n"
  "\n"
  "	v (mV)\n"
  "        ek (mV)		: must be explicitely def. in hoc\n"
  "	celsius		(degC)\n"
  "	gbar=.003 (mho/cm2)\n"
  "        vhalfn = -15: 13 : -25  : -20  (mV)\n"
  "        a0n=0.02      (/ms)\n"
  "        zetan=-3    (1)\n"
  "        gmn=0.7  (1)\n"
  "	nmax=2  (1)\n"
  "	qt=1\n"
  "}\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX kdrDA\n"
  "	USEION k READ ek WRITE ik\n"
  "        RANGE gkdr, i, gbar\n"
  "	RANGE ninf,taun\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	n\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ik (mA/cm2)\n"
  "	i  (mA/cm2)\n"
  "        ninf\n"
  "        gkdr\n"
  "        taun\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	gkdr = gbar*n\n"
  "	ik = gkdr*(v-ek)*DA1(t)*DA2(t)\n"
  "	i = ik\n"
  "\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	n=ninf\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION alpn(v(mV)) {\n"
  "  alpn = exp(1.e-3*(-3)*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) \n"
  "}\n"
  "\n"
  "FUNCTION betn(v(mV)) {\n"
  "  betn = exp(1.e-3*(-3)*(0.7)*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) \n"
  "}\n"
  "\n"
  "DERIVATIVE states {     : exact when v held constant; integrates over dt step\n"
  "        rates(v)\n"
  "        n' = (ninf - n)/taun\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV)) { :callable from hoc\n"
  "        LOCAL a\n"
  "        a = alpn(v)\n"
  "		if (v < -55 ) {              ::::::::::::::::::::   -55\n"
  "		ninf = 0\n"
  "		} else{\n"
  "		ninf = 1 / ( 1 + exp( ( vhalfn - v ) / 11 ) )\n"
  "		:ninf = 1 / ( 1 + exp( ( - v + 13 ) / 8.738 ) )\n"
  "        }\n"
  "		taun = betn(v)/(qt*(0.02)*(1+a))\n"
  "	if (taun<nmax) {taun=nmax}\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION DA1(t) {\n"
  "	    if (t >= DA_start && t <= DA_stop){ 									: During conditioning\n"
  "			if ((t/tone_period-floor(t/tone_period)) >= (1-DA_period/tone_period)) {DA1 = DA_t1}\n"
  "			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA1 = DA_t1}\n"
  "			else {DA1 = 1}}\n"
  "		else if (t >= DA_ext1 && t <= DA_ext2){								: During 4trials of Extinction\n"
  "			if ((t/tone_period-floor(t/tone_period)) >= (1-DA_period/tone_period)) {DA1 = DA_t1}\n"
  "			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA1 = DA_t1}\n"
  "			else {DA1 = 1}}		\n"
  "		else  {DA1 = 1}\n"
  "	}\n"
  "FUNCTION DA2(t) {\n"
  "	    if (t >= DA_start2 && t <= DA_stop){\n"
  "			if((t/tone_period-floor(t/tone_period)) >= (1-DA_period2/tone_period)) {DA2 = DA_t2}\n"
  "			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA2 = DA_t2}\n"
  "			else  {DA2 = 1}}\n"
  "		else  {DA2 = 1}\n"
  "	}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
