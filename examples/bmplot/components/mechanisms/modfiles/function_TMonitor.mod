: passive leak current

NEURON {
	SUFFIX TMonitor
	RANGE tmon,percentage,totaltime
}

UNITS {
	
}

PARAMETER {
	
}

ASSIGNED {
	tmon (ms)
	percentage
	lastpercentage
	totaltime (ms)
}

INITIAL {
	lastpercentage = -1
	percentage = 0
}

BREAKPOINT { 
	tmon = t
	percentage = ceil(t*100.0/totaltime)-1
	if(percentage!=lastpercentage) {
		printf("%f percent is done!\n",percentage)
		lastpercentage = percentage
	}
	
}