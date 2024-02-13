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
 
#define nrn_init _nrn_init__ca
#define _nrn_initial _nrn_initial__ca
#define nrn_cur _nrn_cur__ca
#define _nrn_current _nrn_current__ca
#define nrn_jacob _nrn_jacob__ca
#define nrn_state _nrn_state__ca
#define _net_receive _net_receive__ca 
#define rates rates__ca 
#define states states__ca 
 
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
#define vshift _p[1]
#define vshift_columnindex 1
#define temp _p[2]
#define temp_columnindex 2
#define q10 _p[3]
#define q10_columnindex 3
#define vmin _p[4]
#define vmin_columnindex 4
#define vmax _p[5]
#define vmax_columnindex 5
#define gca _p[6]
#define gca_columnindex 6
#define minf _p[7]
#define minf_columnindex 7
#define hinf _p[8]
#define hinf_columnindex 8
#define mtau _p[9]
#define mtau_columnindex 9
#define htau _p[10]
#define htau_columnindex 10
#define tadj _p[11]
#define tadj_columnindex 11
#define m _p[12]
#define m_columnindex 12
#define h _p[13]
#define h_columnindex 13
#define ica _p[14]
#define ica_columnindex 14
#define eca _p[15]
#define eca_columnindex 15
#define Dm _p[16]
#define Dm_columnindex 16
#define Dh _p[17]
#define Dh_columnindex 17
#define v _p[18]
#define v_columnindex 18
#define _g _p[19]
#define _g_columnindex 19
#define _ion_eca	*(_ppvar[0].get<double*>())
#define _ion_ica	*_ppvar[1].get<double*>()
#define _ion_dicadv	*_ppvar[2].get<double*>()
 
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
 static void _hoc_efun(void);
 static void _hoc_rates(void);
 static void _hoc_states(void);
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
 {"setdata_ca", _hoc_setdata},
 {"efun_ca", _hoc_efun},
 {"rates_ca", _hoc_rates},
 {"states_ca", _hoc_states},
 {0, 0}
};
#define efun efun_ca
 extern double efun( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define cai cai_ca
 double cai = 0;
#define cao cao_ca
 double cao = 2.5;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"cao_ca", "mM"},
 {"cai_ca", "mM"},
 {"gbar_ca", "pS/um2"},
 {"vshift_ca", "mV"},
 {"temp_ca", "degC"},
 {"vmin_ca", "mV"},
 {"vmax_ca", "mV"},
 {"gca_ca", "pS/um2"},
 {"mtau_ca", "ms"},
 {"htau_ca", "ms"},
 {0, 0}
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"cao_ca", &cao_ca},
 {"cai_ca", &cai_ca},
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
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ca",
 "gbar_ca",
 "vshift_ca",
 "temp_ca",
 "q10_ca",
 "vmin_ca",
 "vmax_ca",
 0,
 "gca_ca",
 "minf_ca",
 "hinf_ca",
 "mtau_ca",
 "htau_ca",
 "tadj_ca",
 0,
 "m_ca",
 "h_ca",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 20, _prop);
 	/*initialize range parameters*/
 	gbar = 0.1;
 	vshift = 0;
 	temp = 23;
 	q10 = 2.3;
 	vmin = -120;
 	vmax = 100;
 	_prop->param = _p;
 	_prop->param_size = 20;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0] = &prop_ion->param[0]; /* eca */
 	_ppvar[1] = &prop_ion->param[3]; /* ica */
 	_ppvar[2] = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _ca_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
  hoc_register_prop_size(_mechtype, 20, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ca /home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/ca.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.78e555060882cp+16, 96485.3};
 
#define R _nrnunit_R[_nrnunit_use_legacy_]
static double _nrnunit_R[2] = {0x1.0a1013e8990bep+3, 8.3145};
 
#define PI _nrnunit_PI[_nrnunit_use_legacy_]
static double _nrnunit_PI[2] = {0x1.921fb54442d18p+1, 3.14159};
static int _reset;
static const char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
static int states(_threadargsproto_);
 
static int  states ( _threadargsproto_ ) {
   double _lmexp , _lhexp , _ltinc ;
 rates ( _threadargscomma_ v + vshift ) ;
   tadj = pow( q10 , ( ( celsius - temp ) / 10.0 ) ) ;
   _ltinc = - dt * tadj ;
   _lmexp = 1.0 - exp ( _ltinc / mtau ) ;
   _lhexp = 1.0 - exp ( _ltinc / htau ) ;
   m = m + _lmexp * ( minf - m ) ;
   h = h + _lhexp * ( hinf - h ) ;
    return 0; }
 
static void _hoc_states(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 states ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
static int  rates ( _threadargsprotocomma_ double _lvm ) {
   double _la , _lb ;
 _la = 0.055 * ( - 27.0 - _lvm ) / ( exp ( ( - 27.0 - _lvm ) / 3.8 ) - 1.0 ) ;
   _lb = 0.94 * exp ( ( - 75.0 - _lvm ) / 17.0 ) ;
   mtau = 1.0 / ( _la + _lb ) ;
   minf = _la * mtau ;
   _la = 0.000457 * exp ( ( - 13.0 - _lvm ) / 50.0 ) ;
   _lb = 0.0065 / ( exp ( ( - _lvm - 15.0 ) / 28.0 ) + 1.0 ) ;
   htau = 1.0 / ( _la + _lb ) ;
   hinf = _la * htau ;
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
 
double efun ( _threadargsprotocomma_ double _lz ) {
   double _lefun;
 if ( fabs ( _lz ) < 1e-4 ) {
     _lefun = 1.0 - _lz / 2.0 ;
     }
   else {
     _lefun = _lz / ( exp ( _lz ) - 1.0 ) ;
     }
   
return _lefun;
 }
 
static void _hoc_efun(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  efun ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("ca", "cannot be used with CVODE"); return 0;}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   m = minf ;
   h = hinf ;
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
  eca = _ion_eca;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gca = tadj * gbar * m * m * h ;
   ica = ( 1e-4 ) * gca * ( v - eca ) ;
   }
 _current += ica;

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
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
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
 {  { states(_p, _ppvar, _thread, _nt); }
  } }}

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
    const char* nmodl_filename = "/home/gjgpb9/LargeScaleBLA/components_homogenous/mechanisms/modfiles/ca.mod";
    const char* nmodl_file_text = 
  "\n"
  "COMMENT\n"
  "\n"
  "ca.mod\n"
  "Uses fixed eca instead of GHK eqn\n"
  "\n"
  "HVA Ca current\n"
  "Based on Reuveni, Friedman, Amitai and Gutnick (1993) J. Neurosci. 13:\n"
  "4609-4621.\n"
  "\n"
  "Author: Zach Mainen, Salk Institute, 1994, zach@salk.edu\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX ca\n"
  "	USEION ca READ eca WRITE ica\n"
  "	RANGE m, h, gca, gbar\n"
  "	RANGE minf, hinf, mtau, htau\n"
  "	RANGE q10, temp, tadj, vmin, vmax, vshift\n"
  "	RANGE mexp, hexp\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	gbar = 0.1   	(pS/um2)	: 0.12 mho/cm2\n"
  "	vshift = 0	(mV)		: voltage shift (affects all)\n"
  "\n"
  "	cao  = 2.5	(mM)	        : external ca concentration\n"
  "	cai		(mM)\n"
  "						\n"
  "	temp = 23	(degC)		: original temp \n"
  "	q10  = 2.3			: temperature sensitivity\n"
  "\n"
  "	v 		(mV)\n"
  "	dt		(ms)\n"
  "	celsius		(degC)\n"
  "	vmin = -120	(mV)\n"
  "	vmax = 100	(mV)\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(pS) = (picosiemens)\n"
  "	(um) = (micron)\n"
  "	FARADAY = (faraday) (coulomb)\n"
  "	R = (k-mole) (joule/degC)\n"
  "	PI	= (pi) (1)\n"
  "} \n"
  "\n"
  "ASSIGNED {\n"
  "	ica 		(mA/cm2)\n"
  "	gca		(pS/um2)\n"
  "	eca		(mV)\n"
  "	minf 		hinf\n"
  "	mtau (ms)	htau (ms)\n"
  "	tadj\n"
  "}\n"
  " \n"
  "\n"
  "STATE { m h }\n"
  "\n"
  "INITIAL { \n"
  "	m = minf\n"
  "	h = hinf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "        SOLVE states\n"
  "        gca = tadj*gbar*m*m*h\n"
  "	ica = (1e-4) * gca * (v - eca)\n"
  "} \n"
  "\n"
  "\n"
  "\n"
  "PROCEDURE states() {\n"
  "\n"
  "        LOCAL mexp, hexp, tinc\n"
  "\n"
  "		rates(v+vshift): not consistently executed from here if usetable == 1\n"
  "\n"
  "        tadj = q10^((celsius - temp)/10)\n"
  "        tinc = -dt * tadj\n"
  "\n"
  "        mexp = 1 - exp(tinc/mtau)\n"
  "        hexp = 1 - exp(tinc/htau)\n"
  "\n"
  "        m = m + mexp*(minf-m)\n"
  "        h = h + hexp*(hinf-h)\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "PROCEDURE rates(vm) {  \n"
  "        LOCAL  a, b\n"
  "\n"
  "	a = 0.055*(-27 - vm)/(exp((-27-vm)/3.8) - 1)\n"
  "	b = 0.94*exp((-75-vm)/17)\n"
  "	\n"
  "	mtau = 1/(a+b)\n"
  "	minf = a*mtau\n"
  "\n"
  "		:\"h\" inactivation \n"
  "\n"
  "	a = 0.000457*exp((-13-vm)/50)\n"
  "	b = 0.0065/(exp((-vm-15)/28) + 1)\n"
  "\n"
  "	htau = 1/(a+b)\n"
  "	hinf = a*htau\n"
  "}\n"
  "\n"
  "FUNCTION efun(z) {\n"
  "	if (fabs(z) < 1e-4) {\n"
  "		efun = 1 - z/2\n"
  "	}else{\n"
  "		efun = z/(exp(z) - 1)\n"
  "	}\n"
  "}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
