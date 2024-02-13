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
 
#define nrn_init _nrn_init__na3
#define _nrn_initial _nrn_initial__na3
#define nrn_cur _nrn_cur__na3
#define _nrn_current _nrn_current__na3
#define nrn_jacob _nrn_jacob__na3
#define nrn_state _nrn_state__na3
#define _net_receive _net_receive__na3 
#define states states__na3 
#define trates trates__na3 
 
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
#define thinf _p[1]
#define thinf_columnindex 1
#define qinf _p[2]
#define qinf_columnindex 2
#define ar2 _p[3]
#define ar2_columnindex 3
#define minf _p[4]
#define minf_columnindex 4
#define hinf _p[5]
#define hinf_columnindex 5
#define mtau _p[6]
#define mtau_columnindex 6
#define htau _p[7]
#define htau_columnindex 7
#define sinf _p[8]
#define sinf_columnindex 8
#define taus _p[9]
#define taus_columnindex 9
#define m _p[10]
#define m_columnindex 10
#define h _p[11]
#define h_columnindex 11
#define s _p[12]
#define s_columnindex 12
#define ena _p[13]
#define ena_columnindex 13
#define ina _p[14]
#define ina_columnindex 14
#define thegna _p[15]
#define thegna_columnindex 15
#define tha1 _p[16]
#define tha1_columnindex 16
#define Dm _p[17]
#define Dm_columnindex 17
#define Dh _p[18]
#define Dh_columnindex 18
#define Ds _p[19]
#define Ds_columnindex 19
#define v _p[20]
#define v_columnindex 20
#define _g _p[21]
#define _g_columnindex 21
#define _ion_ena	*(_ppvar[0].get<double*>())
#define _ion_ina	*_ppvar[1].get<double*>()
#define _ion_dinadv	*_ppvar[2].get<double*>()
 
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
 static void _hoc_alps(void);
 static void _hoc_alpv(void);
 static void _hoc_bets(void);
 static void _hoc_trap0(void);
 static void _hoc_trates(void);
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
 {"setdata_na3", _hoc_setdata},
 {"alps_na3", _hoc_alps},
 {"alpv_na3", _hoc_alpv},
 {"bets_na3", _hoc_bets},
 {"trap0_na3", _hoc_trap0},
 {"trates_na3", _hoc_trates},
 {0, 0}
};
#define alps alps_na3
#define alpv alpv_na3
#define bets bets_na3
#define trap0 trap0_na3
 extern double alps( _threadargsprotocomma_ double );
 extern double alpv( _threadargsprotocomma_ double );
 extern double bets( _threadargsprotocomma_ double );
 extern double trap0( _threadargsprotocomma_ double , double , double , double );
 #define _zmexp _thread[0].get<double*>()[0]
 #define _zhexp _thread[0].get<double*>()[1]
 #define _zsexp _thread[0].get<double*>()[2]
 /* declare global and static user variables */
#define Rd Rd_na3
 double Rd = 0.03;
#define Rg Rg_na3
 double Rg = 0.01;
#define Rb Rb_na3
 double Rb = 0.124;
#define Ra Ra_na3
 double Ra = 0.4;
#define a0s a0s_na3
 double a0s = 0.0003;
#define gms gms_na3
 double gms = 0.2;
#define hmin hmin_na3
 double hmin = 0.5;
#define mmin mmin_na3
 double mmin = 0.02;
#define qq qq_na3
 double qq = 10;
#define q10 q10_na3
 double q10 = 2;
#define qg qg_na3
 double qg = 1.5;
#define qd qd_na3
 double qd = 1.5;
#define qa qa_na3
 double qa = 7.2;
#define smax smax_na3
 double smax = 10;
#define tq tq_na3
 double tq = -55;
#define thi2 thi2_na3
 double thi2 = -45;
#define thi1 thi1_na3
 double thi1 = -45;
#define tha tha_na3
 double tha = -30;
#define vvs vvs_na3
 double vvs = 2;
#define vvh vvh_na3
 double vvh = -58;
#define vhalfs vhalfs_na3
 double vhalfs = -60;
#define zetas zetas_na3
 double zetas = 12;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"tha_na3", "mV"},
 {"qa_na3", "mV"},
 {"Ra_na3", "/ms"},
 {"Rb_na3", "/ms"},
 {"thi1_na3", "mV"},
 {"thi2_na3", "mV"},
 {"qd_na3", "mV"},
 {"qg_na3", "mV"},
 {"Rg_na3", "/ms"},
 {"Rd_na3", "/ms"},
 {"qq_na3", "mV"},
 {"tq_na3", "mV"},
 {"vhalfs_na3", "mV"},
 {"a0s_na3", "ms"},
 {"zetas_na3", "1"},
 {"gms_na3", "1"},
 {"smax_na3", "ms"},
 {"vvh_na3", "mV"},
 {"vvs_na3", "mV"},
 {"gbar_na3", "mho/cm2"},
 {"thinf_na3", "mV"},
 {"qinf_na3", "mV"},
 {"ar2_na3", "1"},
 {"mtau_na3", "ms"},
 {"htau_na3", "ms"},
 {"sinf_na3", "ms"},
 {"taus_na3", "ms"},
 {0, 0}
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double s0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"tha_na3", &tha_na3},
 {"qa_na3", &qa_na3},
 {"Ra_na3", &Ra_na3},
 {"Rb_na3", &Rb_na3},
 {"thi1_na3", &thi1_na3},
 {"thi2_na3", &thi2_na3},
 {"qd_na3", &qd_na3},
 {"qg_na3", &qg_na3},
 {"mmin_na3", &mmin_na3},
 {"hmin_na3", &hmin_na3},
 {"q10_na3", &q10_na3},
 {"Rg_na3", &Rg_na3},
 {"Rd_na3", &Rd_na3},
 {"qq_na3", &qq_na3},
 {"tq_na3", &tq_na3},
 {"vhalfs_na3", &vhalfs_na3},
 {"a0s_na3", &a0s_na3},
 {"zetas_na3", &zetas_na3},
 {"gms_na3", &gms_na3},
 {"smax_na3", &smax_na3},
 {"vvh_na3", &vvh_na3},
 {"vvs_na3", &vvs_na3},
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
"na3",
 "gbar_na3",
 "thinf_na3",
 "qinf_na3",
 "ar2_na3",
 0,
 "minf_na3",
 "hinf_na3",
 "mtau_na3",
 "htau_na3",
 "sinf_na3",
 "taus_na3",
 0,
 "m_na3",
 "h_na3",
 "s_na3",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 22, _prop);
 	/*initialize range parameters*/
 	gbar = 0.01;
 	thinf = -50;
 	qinf = 4;
 	ar2 = 1;
 	_prop->param = _p;
 	_prop->param_size = 22;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0] = &prop_ion->param[0]; /* ena */
 	_ppvar[1] = &prop_ion->param[3]; /* ina */
 	_ppvar[2] = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _na3_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 2);
  _extcall_thread = (Datum*)ecalloc(1, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
  hoc_register_prop_size(_mechtype, 22, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 na3 /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/na3.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 /*Top LOCAL _zmexp , _zhexp , _zsexp */
static int _reset;
static const char *modelname = "na3";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int trates(_threadargsprotocomma_ double, double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
double alpv ( _threadargsprotocomma_ double _lv ) {
   double _lalpv;
 _lalpv = 1.0 / ( 1.0 + exp ( ( _lv - vvh ) / vvs ) ) ;
   
return _lalpv;
 }
 
static void _hoc_alpv(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alpv ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double alps ( _threadargsprotocomma_ double _lv ) {
   double _lalps;
 _lalps = exp ( 1.e-3 * zetas * ( _lv - vhalfs ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lalps;
 }
 
static void _hoc_alps(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alps ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double bets ( _threadargsprotocomma_ double _lv ) {
   double _lbets;
 _lbets = exp ( 1.e-3 * zetas * gms * ( _lv - vhalfs ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lbets;
 }
 
static void _hoc_bets(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  bets ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   trates ( _threadargscomma_ v , ar2 ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   Ds = ( sinf - s ) / taus ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 trates ( _threadargscomma_ v , ar2 ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
 Ds = Ds  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taus )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   trates ( _threadargscomma_ v , ar2 ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
    s = s + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taus)))*(- ( ( ( sinf ) ) / taus ) / ( ( ( ( - 1.0 ) ) ) / taus ) - s) ;
   }
  return 0;
}
 
static int  trates ( _threadargsprotocomma_ double _lvm , double _la2 ) {
   double _la , _lb , _lc , _lqt ;
 _lqt = pow( q10 , ( ( celsius - 24.0 ) / 10.0 ) ) ;
   tha1 = tha ;
   _la = trap0 ( _threadargscomma_ _lvm , tha1 , Ra , qa ) ;
   _lb = trap0 ( _threadargscomma_ - _lvm , - tha1 , Rb , qa ) ;
   mtau = 1.0 / ( _la + _lb ) / _lqt ;
   if ( mtau < mmin ) {
     mtau = mmin ;
     }
   if ( v < - 57.5 ) {
     minf = 0.0 ;
     }
   else {
     minf = _la / ( _la + _lb ) ;
     }
   _la = trap0 ( _threadargscomma_ _lvm , thi1 , Rd , qd ) ;
   _lb = trap0 ( _threadargscomma_ - _lvm , - thi2 , Rg , qg ) ;
   htau = 1.0 / ( _la + _lb ) / _lqt ;
   if ( htau < hmin ) {
     htau = hmin ;
     }
   hinf = 1.0 / ( 1.0 + exp ( ( _lvm - thinf ) / qinf ) ) ;
   _lc = alpv ( _threadargscomma_ _lvm ) ;
   sinf = _lc + _la2 * ( 1.0 - _lc ) ;
   taus = bets ( _threadargscomma_ _lvm ) / ( a0s * ( 1.0 + alps ( _threadargscomma_ _lvm ) ) ) ;
   if ( taus < smax ) {
     taus = smax ;
     }
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 trates ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double trap0 ( _threadargsprotocomma_ double _lv , double _lth , double _la , double _lq ) {
   double _ltrap0;
 if ( fabs ( _lv - _lth ) > 1e-6 ) {
     _ltrap0 = _la * ( _lv - _lth ) / ( 1.0 - exp ( - ( _lv - _lth ) / _lq ) ) ;
     }
   else {
     _ltrap0 = _la * _lq ;
     }
   
return _ltrap0;
 }
 
static void _hoc_trap0(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  trap0 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
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
  ena = _ion_ena;
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
  ena = _ion_ena;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
   _thread[0] = (double*)ecalloc(3, sizeof(double));
 }
 
static void _thread_cleanup(Datum* _thread) {
   free(_thread[0].get<void*>());
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
  s = s0;
 {
   trates ( _threadargscomma_ v , ar2 ) ;
   m = minf ;
   h = hinf ;
   s = sinf ;
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
  ena = _ion_ena;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   thegna = gbar * m * m * m * h * s ;
   ina = thegna * ( v - ena ) ;
   }
 _current += ina;

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
  ena = _ion_ena;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
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
  ena = _ion_ena;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
 _slist1[2] = s_columnindex;  _dlist1[2] = Ds_columnindex;
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/na3.mod";
    const char* nmodl_file_text = 
  "TITLE na3\n"
  ": Na current \n"
  ": from Jeff M.\n"
  ":  ---------- modified -------M.Migliore may97\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX na3\n"
  "	USEION na READ ena WRITE ina\n"
  "	RANGE  gbar, ar2\n"
  "	RANGE minf, hinf, mtau, htau, sinf, taus,qinf, thinf\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	\n"
  "	gbar = 0.010   	(mho/cm2)	\n"
  "								\n"
  "	tha  =  -30	(mV)		: v 1/2 for act	\n"
  "	qa   = 7.2	(mV)		: act slope (4.5)		\n"
  "	Ra   = 0.4	(/ms)		: open (v)		\n"
  "	Rb   = 0.124 	(/ms)		: close (v)		\n"
  "\n"
  "	thi1  = -45	(mV)		: v 1/2 for inact 	\n"
  "	thi2  = -45 	(mV)		: v 1/2 for inact 	\n"
  "	qd   = 1.5	(mV)	        : inact tau slope\n"
  "	qg   = 1.5      (mV)\n"
  "	mmin=0.02	\n"
  "	hmin=0.5			\n"
  "	q10=2\n"
  "	Rg   = 0.01 	(/ms)		: inact recov (v) 	\n"
  "	Rd   = .03 	(/ms)		: inact (v)	\n"
  "	qq   = 10        (mV)\n"
  "	tq   = -55      (mV)\n"
  "\n"
  "	thinf  = -50 	(mV)		: inact inf slope	\n"
  "	qinf  = 4 	(mV)		: inact inf slope \n"
  "\n"
  "        vhalfs=-60	(mV)		: slow inact.\n"
  "        a0s=0.0003	(ms)		: a0s=b0s\n"
  "        zetas=12	(1)\n"
  "        gms=0.2		(1)\n"
  "        smax=10		(ms)\n"
  "        vvh=-58		(mV) \n"
  "        vvs=2		(mV)\n"
  "        ar2=1		(1)		: 1=no inact., 0=max inact.\n"
  "	ena		(mV)            : must be explicitly def. in hoc\n"
  "	celsius\n"
  "	v 		(mV)\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(pS) = (picosiemens)\n"
  "	(um) = (micron)\n"
  "} \n"
  "\n"
  "ASSIGNED {\n"
  "	ina 		(mA/cm2)\n"
  "	thegna		(mho/cm2)\n"
  "	minf 		hinf 		\n"
  "	mtau (ms)	htau (ms) 	\n"
  "	sinf (ms)	taus (ms)\n"
  "	tha1	\n"
  "}\n"
  " \n"
  "\n"
  "STATE { m h s}\n"
  "\n"
  "BREAKPOINT {\n"
  "        SOLVE states METHOD cnexp\n"
  "        thegna = gbar*m*m*m*h*s\n"
  "	ina = thegna * (v - ena)\n"
  "} \n"
  "\n"
  "INITIAL {\n"
  "	trates(v,ar2)\n"
  "	m=minf  \n"
  "	h=hinf\n"
  "	s=sinf\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION alpv(v(mV)) {\n"
  "         alpv = 1/(1+exp((v-vvh)/vvs))\n"
  "}\n"
  "        \n"
  "FUNCTION alps(v(mV)) {  \n"
  "  alps = exp(1.e-3*zetas*(v-vhalfs)*9.648e4/(8.315*(273.16+celsius)))\n"
  "}\n"
  "\n"
  "FUNCTION bets(v(mV)) {\n"
  "  bets = exp(1.e-3*zetas*gms*(v-vhalfs)*9.648e4/(8.315*(273.16+celsius)))\n"
  "}\n"
  "\n"
  "LOCAL mexp, hexp, sexp\n"
  "\n"
  "DERIVATIVE states {   \n"
  "        trates(v,ar2)      \n"
  "        m' = (minf-m)/mtau\n"
  "        h' = (hinf-h)/htau\n"
  "        s' = (sinf - s)/taus\n"
  "}\n"
  "\n"
  "PROCEDURE trates(vm,a2) {  \n"
  "        LOCAL  a, b, c, qt\n"
  "        qt=q10^((celsius-24)/10)\n"
  "		tha1 = tha \n"
  "	a = trap0(vm,tha1,Ra,qa)\n"
  "	b = trap0(-vm,-tha1,Rb,qa)\n"
  "	mtau = 1/(a+b)/qt\n"
  "        if (mtau<mmin) {mtau=mmin}\n"
  "	\n"
  "	if (v < -57.5 ) {\n"
  "	minf = 0\n"
  "	} else{\n"
  "	minf = a/(a+b)\n"
  "	}\n"
  "\n"
  "	a = trap0(vm,thi1,Rd,qd)\n"
  "	b = trap0(-vm,-thi2,Rg,qg)\n"
  "	htau =  1/(a+b)/qt\n"
  "        if (htau<hmin) {htau=hmin}\n"
  "	hinf = 1/(1+exp((vm-thinf)/qinf))\n"
  "	c=alpv(vm)\n"
  "        sinf = c+a2*(1-c)\n"
  "        taus = bets(vm)/(a0s*(1+alps(vm)))\n"
  "        if (taus<smax) {taus=smax}\n"
  "}\n"
  "\n"
  "FUNCTION trap0(v,th,a,q) {\n"
  "	if (fabs(v-th) > 1e-6) {\n"
  "	        trap0 = a * (v - th) / (1 - exp(-(v - th)/q))\n"
  "	} else {\n"
  "	        trap0 = a * q\n"
  " 	}\n"
  "}	\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
