: exp2syn1_stsp.mod

NEURON {
  POINT_PROCESS Exp2Syn1_STSP
  RANGE tau, e, i
  NONSPECIFIC_CURRENT i
  
  : short term faciliation and depression
  RANGE F, f, tauF, D1, d1, tauD1, D2, d2, tauD2
  RANGE facfactor
  RANGE initW     : synaptic scaler for large scale networks
}

PARAMETER {
  tau = 0.1 (ms)
  e = 0 (millivolt)

  initW	= 1.0			: synaptic scaler for large scale networks (nS)
  
  : short term faciliation and depression
  facfactor = 1
  : the (1) is needed for the range limits to be effective
  f = 1 (1) < 0, 1e9 >    : facilitation
  tauF = 1 (ms) < 1e-9, 1e9 >
  d1 = 1 (1) < 0, 1 >     : fast depression
  tauD1 = 1 (ms) < 1e-9, 1e9 >
  d2 = 1 (1) < 0, 1 >     : slow depression
  tauD2 = 1 (ms) < 1e-9, 1e9 >
}

ASSIGNED {
  v (millivolt)
  i (nanoamp)
  
  : short term faciliation and depression
  t0 (ms)
  tsyn
  F
  D1
  D2
}

STATE { g (microsiemens) }

INITIAL { 
  g = 0 
   
  : short term facilitation and depression
  t0 = -1
  tsyn = -1e30 
  F = 0
  D1 = 1
  D2 = 1 
}

BREAKPOINT { 
  SOLVE state METHOD cnexp
  i = g*facfactor*(v - e) * initW
}

DERIVATIVE state { g' = -g/tau }

NET_RECEIVE(weight (microsiemens)) {
  
  g = g + weight :Original Exp2Syn1 way
  
  : short term faciliation and depression
  t0 = t
  
  F  = 1 + (F-1)* exp(-(t - tsyn)/tauF)
  D1 = 1 - (1-D1)*exp(-(t - tsyn)/tauD1)
  D2 = 1 - (1-D2)*exp(-(t - tsyn)/tauD2)
  
  tsyn = t
  
  facfactor = F * D1 * D2
  
  F = F * f
  if(F > 30){
    F = 30
  }
  
  D1 = D1 * d1
  D2 = D2 * d2
}
