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
 
#define nrn_init _nrn_init__bg2pyr
#define _nrn_initial _nrn_initial__bg2pyr
#define nrn_cur _nrn_cur__bg2pyr
#define _nrn_current _nrn_current__bg2pyr
#define nrn_jacob _nrn_jacob__bg2pyr
#define nrn_state _nrn_state__bg2pyr
#define _net_receive _net_receive__bg2pyr 
#define states states__bg2pyr 
 
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
#define initW _p[0]
#define initW_columnindex 0
#define taun1 _p[1]
#define taun1_columnindex 1
#define taun2 _p[2]
#define taun2_columnindex 2
#define gNMDAmax _p[3]
#define gNMDAmax_columnindex 3
#define enmda _p[4]
#define enmda_columnindex 4
#define taua1 _p[5]
#define taua1_columnindex 5
#define taua2 _p[6]
#define taua2_columnindex 6
#define gAMPAmax _p[7]
#define gAMPAmax_columnindex 7
#define eampa _p[8]
#define eampa_columnindex 8
#define inmda _p[9]
#define inmda_columnindex 9
#define gnmda _p[10]
#define gnmda_columnindex 10
#define gnmdas _p[11]
#define gnmdas_columnindex 11
#define factorn _p[12]
#define factorn_columnindex 12
#define normconstn _p[13]
#define normconstn_columnindex 13
#define iampa _p[14]
#define iampa_columnindex 14
#define gampa _p[15]
#define gampa_columnindex 15
#define gampas _p[16]
#define gampas_columnindex 16
#define factora _p[17]
#define factora_columnindex 17
#define normconsta _p[18]
#define normconsta_columnindex 18
#define An _p[19]
#define An_columnindex 19
#define Bn _p[20]
#define Bn_columnindex 20
#define Aa _p[21]
#define Aa_columnindex 21
#define Ba _p[22]
#define Ba_columnindex 22
#define eca _p[23]
#define eca_columnindex 23
#define DAn _p[24]
#define DAn_columnindex 24
#define DBn _p[25]
#define DBn_columnindex 25
#define DAa _p[26]
#define DAa_columnindex 26
#define DBa _p[27]
#define DBa_columnindex 27
#define v _p[28]
#define v_columnindex 28
#define _g _p[29]
#define _g_columnindex 29
#define _tsav _p[30]
#define _tsav_columnindex 30
#define _nd_area  *_ppvar[0].get<double*>()
 
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
 static double _hoc_sfunc(void*);
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
 {"sfunc", _hoc_sfunc},
 {0, 0}
};
#define sfunc sfunc_bg2pyr
 extern double sfunc( _threadargsprotocomma_ double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"taun1", "ms"},
 {"taun2", "ms"},
 {"gNMDAmax", "uS"},
 {"enmda", "mV"},
 {"taua1", "ms"},
 {"taua2", "ms"},
 {"gAMPAmax", "uS"},
 {"eampa", "mV"},
 {"inmda", "nA"},
 {"iampa", "nA"},
 {0, 0}
};
 static double Aa0 = 0;
 static double An0 = 0;
 static double Ba0 = 0;
 static double Bn0 = 0;
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
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, Memb_list*, int);
static void _ode_matsol(NrnThread*, Memb_list*, int);
 
#define _cvode_ieq _ppvar[2].literal_value<int>()
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"bg2pyr",
 "initW",
 "taun1",
 "taun2",
 "gNMDAmax",
 "enmda",
 "taua1",
 "taua2",
 "gAMPAmax",
 "eampa",
 0,
 "inmda",
 "gnmda",
 "gnmdas",
 "factorn",
 "normconstn",
 "iampa",
 "gampa",
 "gampas",
 "factora",
 "normconsta",
 0,
 "An",
 "Bn",
 "Aa",
 "Ba",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 31, _prop);
 	/*initialize range parameters*/
 	initW = 6.3;
 	taun1 = 5;
 	taun2 = 125;
 	gNMDAmax = 0.0005;
 	enmda = 0;
 	taua1 = 0.5;
 	taua2 = 7;
 	gAMPAmax = 0.001;
 	eampa = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 31;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _bg2pyr_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
  hoc_register_prop_size(_mechtype, 31, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 bg2pyr /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/bg2pyr.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[4], _dlist1[4];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DAn = - An / taun1 ;
   DBn = - Bn / taun2 ;
   DAa = - Aa / taua1 ;
   DBa = - Ba / taua2 ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DAn = DAn  / (1. - dt*( ( - 1.0 ) / taun1 )) ;
 DBn = DBn  / (1. - dt*( ( - 1.0 ) / taun2 )) ;
 DAa = DAa  / (1. - dt*( ( - 1.0 ) / taua1 )) ;
 DBa = DBa  / (1. - dt*( ( - 1.0 ) / taua2 )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    An = An + (1. - exp(dt*(( - 1.0 ) / taun1)))*(- ( 0.0 ) / ( ( - 1.0 ) / taun1 ) - An) ;
    Bn = Bn + (1. - exp(dt*(( - 1.0 ) / taun2)))*(- ( 0.0 ) / ( ( - 1.0 ) / taun2 ) - Bn) ;
    Aa = Aa + (1. - exp(dt*(( - 1.0 ) / taua1)))*(- ( 0.0 ) / ( ( - 1.0 ) / taua1 ) - Aa) ;
    Ba = Ba + (1. - exp(dt*(( - 1.0 ) / taua2)))*(- ( 0.0 ) / ( ( - 1.0 ) / taua2 ) - Ba) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   double _lx ;
 _lx = _args[0] ;
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = An;
    double __primary = (An + _lx) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / taun1 ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / taun1 ) - __primary );
    An += __primary;
  } else {
 An = An + _lx ;
     }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Bn;
    double __primary = (Bn + _lx) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / taun2 ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / taun2 ) - __primary );
    Bn += __primary;
  } else {
 Bn = Bn + _lx ;
     }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Aa;
    double __primary = (Aa + _lx) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / taua1 ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / taua1 ) - __primary );
    Aa += __primary;
  } else {
 Aa = Aa + _lx ;
     }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = Ba;
    double __primary = (Ba + _lx) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / taua2 ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / taua2 ) - __primary );
    Ba += __primary;
  } else {
 Ba = Ba + _lx ;
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
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  Aa = Aa0;
  An = An0;
  Ba = Ba0;
  Bn = Bn0;
 {
   An = 0.0 ;
   Bn = 0.0 ;
   factorn = taun1 * taun2 / ( taun2 - taun1 ) ;
   normconstn = - 1.0 / ( factorn * ( 1.0 / exp ( log ( taun2 / taun1 ) / ( taun1 * ( 1.0 / taun1 - 1.0 / taun2 ) ) ) - 1.0 / exp ( log ( taun2 / taun1 ) / ( taun2 * ( 1.0 / taun1 - 1.0 / taun2 ) ) ) ) ) ;
   Aa = 0.0 ;
   Ba = 0.0 ;
   factora = taua1 * taua2 / ( taua2 - taua1 ) ;
   normconsta = - 1.0 / ( factora * ( 1.0 / exp ( log ( taua2 / taua1 ) / ( taua1 * ( 1.0 / taua1 - 1.0 / taua2 ) ) ) - 1.0 / exp ( log ( taua2 / taua1 ) / ( taua2 * ( 1.0 / taua1 - 1.0 / taua2 ) ) ) ) ) ;
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gnmda = normconstn * factorn * ( Bn - An ) ;
   gnmdas = gnmda ;
   if ( gnmdas > 1.0 ) {
     gnmdas = 1.0 ;
     }
   inmda = initW * gNMDAmax * gnmdas * ( v - enmda ) * sfunc ( _threadargscomma_ v ) ;
   gampa = normconsta * factora * ( Ba - Aa ) ;
   gampas = gampa ;
   if ( gampas > 1.0 ) {
     gampas = 1.0 ;
     }
   iampa = initW * gAMPAmax * gampas * ( v - eampa ) ;
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
 {   states(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = An_columnindex;  _dlist1[0] = DAn_columnindex;
 _slist1[1] = Bn_columnindex;  _dlist1[1] = DBn_columnindex;
 _slist1[2] = Aa_columnindex;  _dlist1[2] = DAa_columnindex;
 _slist1[3] = Ba_columnindex;  _dlist1[3] = DBa_columnindex;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/bg2pyr.mod";
    const char* nmodl_file_text = 
  ":Background to Pyramidal Cells AMPA+NMDA \n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS bg2pyr\n"
  "	NONSPECIFIC_CURRENT inmda\n"
  "	NONSPECIFIC_CURRENT iampa\n"
  "	RANGE taun1, taun2, factorn, normconstn\n"
  "	RANGE taua1, taua2, factora, normconsta\n"
  "	RANGE gnmda, gnmdas, gNMDAmax, enmda\n"
  "	RANGE gampa, gampas, gAMPAmax, eampa\n"
  "	RANGE initW\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mV) = (millivolt)\n"
  "        (nA) = (nanoamp)\n"
  "	(uS) = (microsiemens)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	:W\n"
  "	initW = 6.3 : 6.3 : 6.3 : 8 :6\n"
  "\n"
  "	:NMDA\n"
  "	taun1 = 5 (ms)\n"
  "	taun2 = 125 (ms)\n"
  "	gNMDAmax = 0.5e-3 (uS)\n"
  "	enmda = 0 (mV)\n"
  "\n"
  "	:AMPA\n"
  "	taua1 = .5 (ms)\n"
  "	taua2 = 7 (ms)\n"
  "	gAMPAmax = 1e-3 (uS)\n"
  "	eampa = 0 (mV)\n"
  "	\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)\n"
  "	eca (mV)\n"
  "	\n"
  "	:NMDA\n"
  "	inmda (nA)\n"
  "	gnmda\n"
  "	gnmdas\n"
  "	factorn\n"
  "	normconstn\n"
  "\n"
  "	:AMPA\n"
  "	iampa (nA)\n"
  "	gampa\n"
  "	gampas\n"
  "	factora\n"
  "	normconsta\n"
  "}\n"
  "\n"
  "STATE {\n"
  "\n"
  "	:NMDA\n"
  "	An\n"
  "	Bn\n"
  "\n"
  "	:AMPA\n"
  "	Aa\n"
  "	Ba\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "\n"
  "	:NMDA\n"
  "	An = 0\n"
  "	Bn = 0\n"
  "	factorn = taun1*taun2/(taun2-taun1)\n"
  "	normconstn = -1/(factorn*(1/exp(log(taun2/taun1)/(taun1*(1/taun1-1/taun2)))-1/exp(log(taun2/taun1)/(taun2*(1/taun1-1/taun2)))))\n"
  "\n"
  "	:AMPA\n"
  "	Aa = 0\n"
  "	Ba = 0\n"
  "	factora = taua1*taua2/(taua2-taua1)\n"
  "	normconsta = -1/(factora*(1/exp(log(taua2/taua1)/(taua1*(1/taua1-1/taua2)))-1/exp(log(taua2/taua1)/(taua2*(1/taua1-1/taua2)))))\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	gnmda = normconstn*factorn*(Bn-An)\n"
  "	gnmdas = gnmda\n"
  "	if (gnmdas>1) {gnmdas=1}\n"
  "	inmda = initW*gNMDAmax*gnmdas*(v-enmda)*sfunc(v)\n"
  "	\n"
  "	gampa = normconsta*factora*(Ba-Aa)\n"
  "	gampas = gampa\n"
  "	if (gampas > 1) {gampas = 1}\n"
  "	iampa = initW*gAMPAmax*gampas*(v-eampa)\n"
  "	\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "\n"
  "	:NMDA\n"
  "	An' = -An/taun1\n"
  "	Bn' = -Bn/taun2\n"
  "	\n"
  "\n"
  "	:AMPA\n"
  "	Aa' = -Aa/taua1\n"
  "	Ba' = -Ba/taua2\n"
  "\n"
  "}\n"
  "\n"
  "NET_RECEIVE(wgt) {\n"
  "	LOCAL x\n"
  "	x = wgt\n"
  "\n"
  "	: state_discontinuity(varname, expression)\n"
  "	: inside a NET_RECEIVE block is equivalent to the assignment statement\n"
  "	: varname = expression\n"
  "	: https://neuron.yale.edu/forum/viewtopic.php?t=1602\n"
  "\n"
  "	An = An+x : state_discontinuity(An,An+x)\n"
  "	Bn = Bn+x : state_discontinuity(Bn,Bn+x)\n"
  "	Aa = Aa+x : state_discontinuity(Aa,Aa+x)\n"
  "	Ba = Ba+x : state_discontinuity(Ba,Ba+x)\n"
  "}\n"
  "\n"
  ":::::::::::: FUNCTIONs and PROCEDUREs ::::::::::::\n"
  "FUNCTION sfunc (v (mV)) {\n"
  "	UNITSOFF\n"
  "	sfunc = 1/(1+0.33*exp(-0.06*v))\n"
  "	UNITSON\n"
  "}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
