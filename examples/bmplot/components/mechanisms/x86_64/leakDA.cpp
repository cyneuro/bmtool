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
 
#define nrn_init _nrn_init__leakDA
#define _nrn_initial _nrn_initial__leakDA
#define nrn_cur _nrn_cur__leakDA
#define _nrn_current _nrn_current__leakDA
#define nrn_jacob _nrn_jacob__leakDA
#define nrn_state _nrn_state__leakDA
#define _net_receive _net_receive__leakDA 
 
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
#define glbar _p[0]
#define glbar_columnindex 0
#define el _p[1]
#define el_columnindex 1
#define il _p[2]
#define il_columnindex 2
#define v _p[3]
#define v_columnindex 3
#define _g _p[4]
#define _g_columnindex 4
 
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
 static void _hoc_DA2(void);
 static void _hoc_DA1(void);
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
 {"setdata_leakDA", _hoc_setdata},
 {"DA2_leakDA", _hoc_DA2},
 {"DA1_leakDA", _hoc_DA1},
 {0, 0}
};
#define DA2 DA2_leakDA
#define DA1 DA1_leakDA
 extern double DA2( _threadargsprotocomma_ double );
 extern double DA1( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define DA_t2 DA_t2_leakDA
 double DA_t2 = 0.9;
#define DA_start2 DA_start2_leakDA
 double DA_start2 = 36000;
#define DA_period2 DA_period2_leakDA
 double DA_period2 = 100;
#define DA_t1 DA_t1_leakDA
 double DA_t1 = 0.8;
#define DA_ext2 DA_ext2_leakDA
 double DA_ext2 = 212000;
#define DA_ext1 DA_ext1_leakDA
 double DA_ext1 = 196000;
#define DA_stop DA_stop_leakDA
 double DA_stop = 96000;
#define DA_start DA_start_leakDA
 double DA_start = 36000;
#define DA_period DA_period_leakDA
 double DA_period = 500;
#define tone_period tone_period_leakDA
 double tone_period = 4000;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"el_leakDA", "mV"},
 {"il_leakDA", "mA/cm2"},
 {0, 0}
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"tone_period_leakDA", &tone_period_leakDA},
 {"DA_period_leakDA", &DA_period_leakDA},
 {"DA_start_leakDA", &DA_start_leakDA},
 {"DA_stop_leakDA", &DA_stop_leakDA},
 {"DA_ext1_leakDA", &DA_ext1_leakDA},
 {"DA_ext2_leakDA", &DA_ext2_leakDA},
 {"DA_t1_leakDA", &DA_t1_leakDA},
 {"DA_period2_leakDA", &DA_period2_leakDA},
 {"DA_start2_leakDA", &DA_start2_leakDA},
 {"DA_t2_leakDA", &DA_t2_leakDA},
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
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"leakDA",
 "glbar_leakDA",
 "el_leakDA",
 0,
 "il_leakDA",
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 5, _prop);
 	/*initialize range parameters*/
 	glbar = 2.85714e-05;
 	el = -75;
 	_prop->param = _p;
 	_prop->param_size = 5;
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _leakDA_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
  hoc_register_prop_size(_mechtype, 5, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 leakDA /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/leakDA.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
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

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{

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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   il = glbar * ( v - el ) * DA1 ( _threadargscomma_ t ) * DA2 ( _threadargscomma_ t ) ;
   }
 _current += il;

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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
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

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/leakDA.mod";
    const char* nmodl_file_text = 
  ": passive leak current\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX leakDA\n"
  "	NONSPECIFIC_CURRENT il\n"
  "	RANGE il, el, glbar\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	tone_period = 4000    \n"
  "	DA_period = 500	\n"
  "	DA_start = 36000		             : D2R(High Affinity) Dopamine Effect after 6 conditioning trials (15*4000) = 60000)\n"
  "	DA_stop = 96000\n"
  "	DA_ext1 = 196000\n"
  "	DA_ext2 = 212000	\n"
  "	DA_t1 = 0.8 : 0.9 : 1 :  1 : 0.9           : Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP\n"
  "	DA_period2 = 100\n"
  "	DA_start2 = 36000		   			: shock Dopamine Effect during shock after 1 conditioning trial\n"
  "	DA_t2 = .9           				: Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP	\n"
  "	\n"
  "	glbar = 2.857142857142857e-05  :3.333333e-5 (siemens/cm2) < 0, 1e9 >\n"
  "	el = -75 (mV)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)\n"
  "	il (mA/cm2)\n"
  "}\n"
  "\n"
  "BREAKPOINT { \n"
  "	il = glbar*(v - el)*DA1(t)*DA2(t)\n"
  "}\n"
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
