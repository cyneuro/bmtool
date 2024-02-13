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
 
#define nrn_init _nrn_init__AMPA_NMDA_STP_PN2SOM
#define _nrn_initial _nrn_initial__AMPA_NMDA_STP_PN2SOM
#define nrn_cur _nrn_cur__AMPA_NMDA_STP_PN2SOM
#define _nrn_current _nrn_current__AMPA_NMDA_STP_PN2SOM
#define nrn_jacob _nrn_jacob__AMPA_NMDA_STP_PN2SOM
#define nrn_state _nrn_state__AMPA_NMDA_STP_PN2SOM
#define _net_receive _net_receive__AMPA_NMDA_STP_PN2SOM 
#define state state__AMPA_NMDA_STP_PN2SOM 
 
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
#define tau_r_AMPA _p[1]
#define tau_r_AMPA_columnindex 1
#define tau_d_AMPA _p[2]
#define tau_d_AMPA_columnindex 2
#define tau_r_NMDA _p[3]
#define tau_r_NMDA_columnindex 3
#define tau_d_NMDA _p[4]
#define tau_d_NMDA_columnindex 4
#define Use _p[5]
#define Use_columnindex 5
#define Dep _p[6]
#define Dep_columnindex 6
#define Fac _p[7]
#define Fac_columnindex 7
#define e _p[8]
#define e_columnindex 8
#define mg _p[9]
#define mg_columnindex 9
#define gmax_NMDA _p[10]
#define gmax_NMDA_columnindex 10
#define gmax_AMPA _p[11]
#define gmax_AMPA_columnindex 11
#define u0 _p[12]
#define u0_columnindex 12
#define NMDA_ratio _p[13]
#define NMDA_ratio_columnindex 13
#define synapseID _p[14]
#define synapseID_columnindex 14
#define verboseLevel _p[15]
#define verboseLevel_columnindex 15
#define conductance _p[16]
#define conductance_columnindex 16
#define record_Pr _p[17]
#define record_Pr_columnindex 17
#define record_use _p[18]
#define record_use_columnindex 18
#define i _p[19]
#define i_columnindex 19
#define i_AMPA _p[20]
#define i_AMPA_columnindex 20
#define i_NMDA _p[21]
#define i_NMDA_columnindex 21
#define g_AMPA _p[22]
#define g_AMPA_columnindex 22
#define g_NMDA _p[23]
#define g_NMDA_columnindex 23
#define g _p[24]
#define g_columnindex 24
#define A_AMPA _p[25]
#define A_AMPA_columnindex 25
#define B_AMPA _p[26]
#define B_AMPA_columnindex 26
#define A_NMDA _p[27]
#define A_NMDA_columnindex 27
#define B_NMDA _p[28]
#define B_NMDA_columnindex 28
#define factor_AMPA _p[29]
#define factor_AMPA_columnindex 29
#define factor_NMDA _p[30]
#define factor_NMDA_columnindex 30
#define mggate _p[31]
#define mggate_columnindex 31
#define DA_AMPA _p[32]
#define DA_AMPA_columnindex 32
#define DB_AMPA _p[33]
#define DB_AMPA_columnindex 33
#define DA_NMDA _p[34]
#define DA_NMDA_columnindex 34
#define DB_NMDA _p[35]
#define DB_NMDA_columnindex 35
#define v _p[36]
#define v_columnindex 36
#define _g _p[37]
#define _g_columnindex 37
#define _tsav _p[38]
#define _tsav_columnindex 38
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
 static double _hoc_toggleVerbose(void*);
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
 {"toggleVerbose", _hoc_toggleVerbose},
 {0, 0}
};
#define toggleVerbose toggleVerbose_AMPA_NMDA_STP_PN2SOM
 extern double toggleVerbose( _threadargsproto_ );
 /* declare global and static user variables */
#define nc_type_param nc_type_param_AMPA_NMDA_STP_PN2SOM
 double nc_type_param = 7;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"tau_r_AMPA", "ms"},
 {"tau_d_AMPA", "ms"},
 {"tau_r_NMDA", "ms"},
 {"tau_d_NMDA", "ms"},
 {"Use", "1"},
 {"Dep", "ms"},
 {"Fac", "ms"},
 {"e", "mV"},
 {"mg", "mM"},
 {"gmax_NMDA", "uS"},
 {"NMDA_ratio", "1"},
 {"i", "nA"},
 {"i_AMPA", "nA"},
 {"i_NMDA", "nA"},
 {"g_AMPA", "uS"},
 {"g_NMDA", "uS"},
 {"g", "uS"},
 {0, 0}
};
 static double A_NMDA0 = 0;
 static double A_AMPA0 = 0;
 static double B_NMDA0 = 0;
 static double B_AMPA0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"nc_type_param_AMPA_NMDA_STP_PN2SOM", &nc_type_param_AMPA_NMDA_STP_PN2SOM},
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
"AMPA_NMDA_STP_PN2SOM",
 "initW",
 "tau_r_AMPA",
 "tau_d_AMPA",
 "tau_r_NMDA",
 "tau_d_NMDA",
 "Use",
 "Dep",
 "Fac",
 "e",
 "mg",
 "gmax_NMDA",
 "gmax_AMPA",
 "u0",
 "NMDA_ratio",
 "synapseID",
 "verboseLevel",
 "conductance",
 "record_Pr",
 "record_use",
 0,
 "i",
 "i_AMPA",
 "i_NMDA",
 "g_AMPA",
 "g_NMDA",
 "g",
 0,
 "A_AMPA",
 "B_AMPA",
 "A_NMDA",
 "B_NMDA",
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
 	_p = nrn_prop_data_alloc(_mechtype, 39, _prop);
 	/*initialize range parameters*/
 	initW = 1;
 	tau_r_AMPA = 0.43;
 	tau_d_AMPA = 10.8;
 	tau_r_NMDA = 3.7;
 	tau_d_NMDA = 125;
 	Use = 0.05;
 	Dep = 800;
 	Fac = 10;
 	e = 0;
 	mg = 1;
 	gmax_NMDA = 0.001;
 	gmax_AMPA = 0.001;
 	u0 = 0;
 	NMDA_ratio = 0.71;
 	synapseID = 0;
 	verboseLevel = 0;
 	conductance = 0;
 	record_Pr = 0;
 	record_use = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 39;
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
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _AMPA_NMDA_STP_PN2SOM_reg() {
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
  hoc_register_prop_size(_mechtype, 39, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 8;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 AMPA_NMDA_STP_PN2SOM /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/AMPA_NMDA_STP_PN2SOM.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "AMPA and NMDA receptor with presynaptic short-term plasticity";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[4], _dlist1[4];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DA_AMPA = - A_AMPA / tau_r_AMPA ;
   DB_AMPA = - B_AMPA / tau_d_AMPA ;
   DA_NMDA = - A_NMDA / tau_r_NMDA ;
   DB_NMDA = - B_NMDA / tau_d_NMDA ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DA_AMPA = DA_AMPA  / (1. - dt*( ( - 1.0 ) / tau_r_AMPA )) ;
 DB_AMPA = DB_AMPA  / (1. - dt*( ( - 1.0 ) / tau_d_AMPA )) ;
 DA_NMDA = DA_NMDA  / (1. - dt*( ( - 1.0 ) / tau_r_NMDA )) ;
 DB_NMDA = DB_NMDA  / (1. - dt*( ( - 1.0 ) / tau_d_NMDA )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    A_AMPA = A_AMPA + (1. - exp(dt*(( - 1.0 ) / tau_r_AMPA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_r_AMPA ) - A_AMPA) ;
    B_AMPA = B_AMPA + (1. - exp(dt*(( - 1.0 ) / tau_d_AMPA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_d_AMPA ) - B_AMPA) ;
    A_NMDA = A_NMDA + (1. - exp(dt*(( - 1.0 ) / tau_r_NMDA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_r_NMDA ) - A_NMDA) ;
    B_NMDA = B_NMDA + (1. - exp(dt*(( - 1.0 ) / tau_d_NMDA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_d_NMDA ) - B_NMDA) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   double _lresult ;
 _args[1] = _args[0] ;
   _args[2] = _args[0] * NMDA_ratio ;
   if ( t < 0.0 ) {
     
/*VERBATIM*/
        return;
 }
   if ( _lflag  == 1.0 ) {
     _args[0] = conductance ;
     }
   if ( Fac > 0.0 ) {
     _args[5] = _args[5] * exp ( - ( t - _args[6] ) / Fac ) ;
     }
   else {
     _args[5] = Use ;
     }
   if ( Fac > 0.0 ) {
     _args[5] = _args[5] + Use * ( 1.0 - _args[5] ) ;
     }
   _args[3] = 1.0 - ( 1.0 - _args[3] ) * exp ( - ( t - _args[6] ) / Dep ) ;
   _args[4] = _args[5] * _args[3] ;
   _args[3] = _args[3] - _args[5] * _args[3] ;
   _args[6] = t ;
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = A_AMPA;
    double __primary = (A_AMPA + _args[4] * _args[1] * factor_AMPA) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_r_AMPA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_r_AMPA ) - __primary );
    A_AMPA += __primary;
  } else {
 A_AMPA = A_AMPA + _args[4] * _args[1] * factor_AMPA ;
     }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = B_AMPA;
    double __primary = (B_AMPA + _args[4] * _args[1] * factor_AMPA) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_d_AMPA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_d_AMPA ) - __primary );
    B_AMPA += __primary;
  } else {
 B_AMPA = B_AMPA + _args[4] * _args[1] * factor_AMPA ;
     }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = A_NMDA;
    double __primary = (A_NMDA + _args[4] * _args[2] * factor_NMDA) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_r_NMDA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_r_NMDA ) - __primary );
    A_NMDA += __primary;
  } else {
 A_NMDA = A_NMDA + _args[4] * _args[2] * factor_NMDA ;
     }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = B_NMDA;
    double __primary = (B_NMDA + _args[4] * _args[2] * factor_NMDA) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_d_NMDA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_d_NMDA ) - __primary );
    B_NMDA += __primary;
  } else {
 B_NMDA = B_NMDA + _args[4] * _args[2] * factor_NMDA ;
     }
 record_use = _args[5] ;
   record_Pr = _args[4] ;
   if ( verboseLevel > 0.0 ) {
     printf ( "value of u %g" , _args[5] ) ;
     }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       double* _p = _pnt->_prop->param;
    Datum* _ppvar = _pnt->_prop->dparam;
    Datum* _thread = (Datum*)0;
    NrnThread* _nt = (NrnThread*)_pnt->_vnt;
 _args[3] = 1.0 ;
   _args[5] = u0 ;
   _args[6] = t ;
   }
 
double toggleVerbose ( _threadargsproto_ ) {
   double _ltoggleVerbose;
 verboseLevel = 1.0 - verboseLevel ;
   
return _ltoggleVerbose;
 }
 
static double _hoc_toggleVerbose(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  toggleVerbose ( _p, _ppvar, _thread, _nt );
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
  A_NMDA = A_NMDA0;
  A_AMPA = A_AMPA0;
  B_NMDA = B_NMDA0;
  B_AMPA = B_AMPA0;
 {
   double _ltp_AMPA , _ltp_NMDA ;
 A_AMPA = 0.0 ;
   B_AMPA = 0.0 ;
   A_NMDA = 0.0 ;
   B_NMDA = 0.0 ;
   _ltp_AMPA = ( tau_r_AMPA * tau_d_AMPA ) / ( tau_d_AMPA - tau_r_AMPA ) * log ( tau_d_AMPA / tau_r_AMPA ) ;
   _ltp_NMDA = ( tau_r_NMDA * tau_d_NMDA ) / ( tau_d_NMDA - tau_r_NMDA ) * log ( tau_d_NMDA / tau_r_NMDA ) ;
   factor_AMPA = - exp ( - _ltp_AMPA / tau_r_AMPA ) + exp ( - _ltp_AMPA / tau_d_AMPA ) ;
   factor_AMPA = 1.0 / factor_AMPA ;
   factor_NMDA = - exp ( - _ltp_NMDA / tau_r_NMDA ) + exp ( - _ltp_NMDA / tau_d_NMDA ) ;
   factor_NMDA = 1.0 / factor_NMDA ;
   record_use = 0.0 ;
   record_Pr = 0.0 ;
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
   mggate = 1.0 / ( 1.0 + exp ( 0.062 * - ( v ) ) * ( mg / 3.57 ) ) ;
   g_AMPA = gmax_AMPA * ( B_AMPA - A_AMPA ) ;
   g_NMDA = gmax_NMDA * ( B_NMDA - A_NMDA ) * mggate ;
   g = g_AMPA + g_NMDA ;
   i_AMPA = g_AMPA * ( v - e ) * initW ;
   i_NMDA = g_NMDA * ( v - e ) * initW ;
   i = i_AMPA + i_NMDA ;
   }
 _current += i;

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
 {   state(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = A_AMPA_columnindex;  _dlist1[0] = DA_AMPA_columnindex;
 _slist1[1] = B_AMPA_columnindex;  _dlist1[1] = DB_AMPA_columnindex;
 _slist1[2] = A_NMDA_columnindex;  _dlist1[2] = DA_NMDA_columnindex;
 _slist1[3] = B_NMDA_columnindex;  _dlist1[3] = DB_NMDA_columnindex;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/AMPA_NMDA_STP_PN2SOM.mod";
    const char* nmodl_file_text = 
  "COMMENT\n"
  "/**\n"
  " * @file DetAMPANMDA.mod\n"
  " * @brief Adapted from ProbAMPANMDA_EMS.mod by Eilif, Michael and Srikanth\n"
  " * @author chindemi\n"
  " * @date 2014-05-25\n"
  " * @remark Copyright (c) BBP/EPFL 2005-2021. This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)\n"
  "* Several changes have been made from the orginal version of this synapse by Greg Glickert to better adapt the model for Large Scale BMTk/Neuron models\n"
  " */\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "TITLE AMPA and NMDA receptor with presynaptic short-term plasticity\n"
  "\n"
  "\n"
  "COMMENT\n"
  "AMPA and NMDA receptor conductance using a dual-exponential profile\n"
  "presynaptic short-term plasticity based on Fuhrmann et al. 2002, deterministic\n"
  "version.\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "NEURON {\n"
  "    THREADSAFE\n"
  "\n"
  "    POINT_PROCESS AMPA_NMDA_STP_PN2SOM\n"
  "    RANGE initW     : synaptic scaler for large scale networks added by Greg\n"
  "    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA\n"
  "    RANGE Use, u, Dep, Fac, u0, mg, NMDA_ratio\n"
  "    RANGE i, i_AMPA, i_NMDA, g_AMPA, g_NMDA, g, e\n"
  "    RANGE gmax_AMPA, gmax_NMDA\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE synapseID, verboseLevel\n"
  "    RANGE conductance\n"
  "    GLOBAL nc_type_param\n"
  "    : For debugging\n"
  "    :RANGE sgid, tgid\n"
  "    RANGE record_Pr, record_use\n"
  "}\n"
  "\n"
  "\n"
  "PARAMETER {\n"
  "    initW         = 1.0      : added by Greg Glickert to scale synaptic weight for large scale modeling\n"
  "    tau_r_AMPA = 0.43   (ms)  : Dual-exponential conductance profile\n"
  "    tau_d_AMPA = 10.8   (ms)  : IMPORTANT: tau_r < tau_d\n"
  "    tau_r_NMDA = 3.7  (ms)  : Dual-exponential conductance profile\n"
  "    tau_d_NMDA = 125    (ms)  : IMPORTANT: tau_r < tau_d\n"
  "    Use = 0.05         (1)   : Utilization of synaptic efficacy\n"
  "    Dep = 800          (ms)  : Relaxation time constant from depression\n"
  "    Fac = 10           (ms)  : Relaxation time constant from facilitation\n"
  "    e = 0              (mV)  : AMPA and NMDA reversal potential\n"
  "    mg = 1             (mM)  : Initial concentration of mg2+\n"
  "    gmax_NMDA = .001        (uS)  : Weight conversion factor (from nS to uS)\n"
  "    gmax_AMPA = .001\n"
  "    u0 = 0                   : Initial value of u, which is the running value of Use\n"
  "    NMDA_ratio = 0.71  (1)   : The ratio of NMDA to AMPA\n"
  "    synapseID = 0\n"
  "    verboseLevel = 0\n"
  "    conductance = 0.0\n"
  "    nc_type_param = 7\n"
  "    record_Pr = 0\n"
  "    record_use = 0\n"
  "    :sgid = -1\n"
  "    :tgid = -1\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "    v (mV)\n"
  "    i (nA)\n"
  "    i_AMPA (nA)\n"
  "    i_NMDA (nA)\n"
  "    g_AMPA (uS)\n"
  "    g_NMDA (uS)\n"
  "    g (uS)\n"
  "    factor_AMPA\n"
  "    factor_NMDA\n"
  "    mggate\n"
  "}\n"
  "\n"
  "\n"
  "STATE {\n"
  "    A_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_r_AMPA\n"
  "    B_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_d_AMPA\n"
  "    A_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_r_NMDA\n"
  "    B_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_d_NMDA\n"
  "    \n"
  "}\n"
  "\n"
  "\n"
  "INITIAL{\n"
  "    LOCAL tp_AMPA, tp_NMDA\n"
  "\n"
  "    A_AMPA = 0\n"
  "    B_AMPA = 0\n"
  "\n"
  "    A_NMDA = 0\n"
  "    B_NMDA = 0\n"
  "\n"
  "    tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA) :time to peak of the conductance\n"
  "    tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA) :time to peak of the conductance\n"
  "\n"
  "    factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA) :AMPA Normalization factor - so that when t = tp_AMPA, gsyn = gpeak\n"
  "    factor_AMPA = 1/factor_AMPA\n"
  "\n"
  "    factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA) :NMDA Normalization factor - so that when t = tp_NMDA, gsyn = gpeak\n"
  "    factor_NMDA = 1/factor_NMDA\n"
  "\n"
  "    record_use = 0\n"
  "    record_Pr = 0\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE state METHOD cnexp\n"
  "    mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 3.57 (mM))) :mggate kinetics - Jahr & Stevens 1990\n"
  "    g_AMPA = gmax_AMPA*(B_AMPA-A_AMPA) :compute time varying conductance as the difference of state variables B_AMPA and A_AMPA\n"
  "    g_NMDA = gmax_NMDA*(B_NMDA-A_NMDA) * mggate :compute time varying conductance as the difference of state variables B_NMDA and A_NMDA and mggate kinetics\n"
  "    g = g_AMPA + g_NMDA\n"
  "    i_AMPA = g_AMPA*(v-e) * initW :compute the AMPA driving force based on the time varying conductance, membrane potential, and AMPA reversal\n"
  "    i_NMDA = g_NMDA*(v-e) * initW :compute the NMDA driving force based on the time varying conductance, membrane potential, and NMDA reversal\n"
  "    i = i_AMPA + i_NMDA\n"
  "}\n"
  "\n"
  "\n"
  "DERIVATIVE state{\n"
  "    A_AMPA' = -A_AMPA/tau_r_AMPA\n"
  "    B_AMPA' = -B_AMPA/tau_d_AMPA\n"
  "    A_NMDA' = -A_NMDA/tau_r_NMDA\n"
  "    B_NMDA' = -B_NMDA/tau_d_NMDA\n"
  "}\n"
  "\n"
  "\n"
  "NET_RECEIVE (weight, weight_AMPA, weight_NMDA, R, Pr, u, tsyn (ms), nc_type){\n"
  "    LOCAL result\n"
  "    weight_AMPA = weight\n"
  "    weight_NMDA = weight * NMDA_ratio\n"
  "\n"
  "    INITIAL{\n"
  "        R=1\n"
  "        u=u0\n"
  "        tsyn=t\n"
  "    }\n"
  "\n"
  "    : Disable in case of t < 0 (in case of ForwardSkip) which causes numerical\n"
  "    : instability if synapses are activated.\n"
  "    if(t < 0 ) {\n"
  "    VERBATIM\n"
  "        return;\n"
  "    ENDVERBATIM\n"
  "    }\n"
  "\n"
  "    if (flag == 1) {\n"
  "        : self event to set next weight at delay\n"
  "          weight = conductance\n"
  "\n"
  "    }\n"
  "    : flag == 0, i.e. a spike has arrived\n"
  "\n"
  "    : calc u at event-\n"
  "    if (Fac > 0) {\n"
  "        u = u*exp(-(t - tsyn)/Fac) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.\n"
  "    } else {\n"
  "        u = Use\n"
  "    }\n"
  "    if(Fac > 0){\n"
  "        u = u + Use*(1-u) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.\n"
  "    }\n"
  "\n"
  "    R  = 1 - (1-R) * exp(-(t-tsyn)/Dep) :Probability R for a vesicle to be available for release, analogous to the pool of synaptic\n"
  "                                        :resources available for release in the deterministic model. Eq. 3 in Fuhrmann et al.\n"
  "    Pr  = u * R                         :Pr is calculated as R * u (running value of Use)\n"
  "    R  = R - u * R                      :update R as per Eq. 3 in Fuhrmann et al.\n"
  "    \n"
  "    \n"
  "    :if( verboseLevel > 0 ) {\n"
  "        :printf(\"Synapse %f at time %g: R = %g Pr = %g erand = %g\\n\", synapseID, t, R, Pr, result )\n"
  "    :}\n"
  "\n"
  "    tsyn = t\n"
  "\n"
  "    A_AMPA = A_AMPA + Pr*weight_AMPA*factor_AMPA\n"
  "    B_AMPA = B_AMPA + Pr*weight_AMPA*factor_AMPA\n"
  "    A_NMDA = A_NMDA + Pr*weight_NMDA*factor_NMDA\n"
  "    B_NMDA = B_NMDA + Pr*weight_NMDA*factor_NMDA\n"
  "\n"
  "    record_use = u\n"
  "    record_Pr = Pr\n"
  "\n"
  "    if( verboseLevel > 0 ) {\n"
  "        printf(\"value of u %g\",u )\n"
  "        :printf( \" vals %g %g %g %g\\n\", A_AMPA, weight_AMPA, factor_AMPA, weight )\n"
  "    }\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION toggleVerbose() {\n"
  "    verboseLevel = 1-verboseLevel\n"
  "}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
