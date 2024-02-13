: passive leak current

NEURON {
	SUFFIX leakDA
	NONSPECIFIC_CURRENT il
	RANGE il, el, glbar
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	tone_period = 4000    
	DA_period = 500	
	DA_start = 36000		             : D2R(High Affinity) Dopamine Effect after 6 conditioning trials (15*4000) = 60000)
	DA_stop = 96000
	DA_ext1 = 196000
	DA_ext2 = 212000	
	DA_t1 = 0.8 : 0.9 : 1 :  1 : 0.9           : Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP
	DA_period2 = 100
	DA_start2 = 36000		   			: shock Dopamine Effect during shock after 1 conditioning trial
	DA_t2 = .9           				: Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP	
	
	glbar = 2.857142857142857e-05  :3.333333e-5 (siemens/cm2) < 0, 1e9 >
	el = -75 (mV)
}

ASSIGNED {
	v (mV)
	il (mA/cm2)
}

BREAKPOINT { 
	il = glbar*(v - el)*DA1(t)*DA2(t)
}
FUNCTION DA1(t) {
	    if (t >= DA_start && t <= DA_stop){ 									: During conditioning
			if ((t/tone_period-floor(t/tone_period)) >= (1-DA_period/tone_period)) {DA1 = DA_t1}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA1 = DA_t1}
			else {DA1 = 1}}
		else if (t >= DA_ext1 && t <= DA_ext2){								: During 4trials of Extinction
			if ((t/tone_period-floor(t/tone_period)) >= (1-DA_period/tone_period)) {DA1 = DA_t1}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA1 = DA_t1}
			else {DA1 = 1}}		
		else  {DA1 = 1}
	}
FUNCTION DA2(t) {
	    if (t >= DA_start2 && t <= DA_stop){
			if((t/tone_period-floor(t/tone_period)) >= (1-DA_period2/tone_period)) {DA2 = DA_t2}
			else if ((t/tone_period-floor(t/tone_period)) == 0) {DA2 = DA_t2}
			else  {DA2 = 1}}
		else  {DA2 = 1}
	}