COMMENT
Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak conductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 is very small compared to tau1, this is an alphasynapse with time constant tau2.
If tau1/tau2 is very small, this is single exponential decay with time constant tau2.

The factor is evaluated in the initial block 
such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

Short-term plasticity based on Fuhrmann et al. 2002, deterministic version.

Caveat: Multiple sources connecting to the same synapse are identified as
the same source, i.e., individual sources share the same set of short-term
plasticity dynamical variables, tsyn, R, u, and Pr.

ENDCOMMENT

NEURON {
	THREADSAFE
	
	POINT_PROCESS Exp2Syn_STP
	RANGE initW     : synaptic scaler for large scale networks
	RANGE tau1, tau2, e, i
	NONSPECIFIC_CURRENT i

	RANGE g
	RANGE Use, Dep, Fac, u0
	RANGE tsyn, R, u, Pr
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau1 = 0.1 (ms) <1e-9,1e9>
	tau2 = 10 (ms) <1e-9,1e9>
	e=0	(mV)

	initW	= 1.0			: synaptic scaler for large scale networks (nS)
	gmax	= 0.001	(uS)	: weight conversion factor (from nS to uS)
	Use		= 1.0	(1)		: Utilization of synaptic efficacy
	Dep		= 0		(ms)	: relaxation time constant from depression
	Fac		= 0		(ms)	: relaxation time constant from facilitation
	u0		= 0				: initial value of u, which is the running value of release probability
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	factor
	tsyn (ms)
	R
	u
	Pr
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	if (tau1/tau2 > 0.9999) {
		tau1 = 0.9999*tau2
	}
	if (tau1/tau2 < 1e-9) {
		tau1 = tau2*1e-9
	}
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = initW*gmax/factor

	tsyn = 0
	R = 1
	u = u0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	i = g*(v - e)
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

NET_RECEIVE(weight (uS)) {
	INITIAL{
		tsyn = t
	}

	: Disable in case of t < 0 (in case of ForwardSkip) which causes numerical
	: instability if synapses are activated.
	if(t < 0 ) {
		VERBATIM
			return;
		ENDVERBATIM
	}

	: calc u at event-
	if (Fac > 0) {
		: update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
		u = u*exp(-(t - tsyn)/Fac)
		u = u + Use*(1-u)
	} else {
		u = Use
	}

	if (Dep > 0) {
		: Probability R for a vesicle to be available for release, analogous to the pool of synaptic
		: resources available for release in the deterministic model. Eq. 3 in Fuhrmann et al.
		R  = 1 - (1-R) * exp(-(t-tsyn)/Dep)
		Pr = u * R			: Pr is calculated as R * u (running value of Use)
		R  = R - u * R		: update R as per Eq. 3 in Fuhrmann et al.
	} else {
		Pr = u 
	}

	tsyn = t

	A = A + Pr*weight*factor
	B = B + Pr*weight*factor
}
