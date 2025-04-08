COMMENT
/**
 * @file DetGABAAB.mod
 * @brief Adapted from ProbGABAA_EMS.mod by Eilif, Michael and Srikanth
 * @author chindemi
 * @date 2014-05-25
 * @remark Copyright (c) BBP/EPFL 2005-2021. This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
 * Several changes have been made from the orginal version of this synapse by Greg Glickert to better adapt the model for Large Scale BMTk/Neuron models
 */
ENDCOMMENT


TITLE GABAA receptor with presynaptic short-term plasticity


COMMENT
GABAA receptor conductance using a dual-exponential profile
presynaptic short-term plasticity based on Fuhrmann et al. 2002, deterministic
version.
ENDCOMMENT


NEURON {
    THREADSAFE

    POINT_PROCESS GABA_A_STP
    RANGE initW     : synaptic scaler for large scale networks added by Greg
    RANGE tau_r_GABAA, tau_d_GABAA
    RANGE Use, Dep, Fac, u0
    RANGE gmax, gmax_GABAA
    RANGE i, g, e_GABAA
    NONSPECIFIC_CURRENT i
    RANGE synapseID, verboseLevel
    RANGE conductance
    GLOBAL nc_type_param
	RANGE record_use, record_Pr
}


PARAMETER {
    initW        = 1.0        : added by Greg Glickert to scale synaptic weight for large scale modeling
    tau_r_GABAA  = 0.2   (ms) : dual-exponential conductance profile
    tau_d_GABAA  = 8     (ms) : IMPORTANT: tau_r < tau_d
    Use          = 1.0   (1)  : Utilization of synaptic efficacy
    Dep          = 100   (ms) : relaxation time constant from depression
    Fac          = 10    (ms) :  relaxation time constant from facilitation
    e_GABAA      = -75   (mV) : GABAA reversal potential was -80mv change to -75 never heard of e_gaba not -75 - Greg
    gmax         = .001  (uS) : weight conversion factor (from nS to uS)
    u0           = 0          :initial value of u, which is the running value of release probability
    synapseID    = 0
    verboseLevel = 0
    conductance  = 0.0
    nc_type_param = 7
}


ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    gmax_GABAA (uS)
    factor_GABAA
    record_use
    record_Pr
}



STATE {
    A_GABAA       : GABAA state variable to construct the dual-exponential profile - decays with conductance tau_r_GABAA
    B_GABAA       : GABAA state variable to construct the dual-exponential profile - decays with conductance tau_d_GABAA
}


INITIAL{
    LOCAL tp_GABAA

    A_GABAA = 0
    B_GABAA = 0

    tp_GABAA = (tau_r_GABAA*tau_d_GABAA)/(tau_d_GABAA-tau_r_GABAA)*log(tau_d_GABAA/tau_r_GABAA) :time to peak of the conductance

    factor_GABAA = -exp(-tp_GABAA/tau_r_GABAA)+exp(-tp_GABAA/tau_d_GABAA) :GABAA Normalization factor - so that when t = tp_GABAA, gsyn = gpeak
    factor_GABAA = 1/factor_GABAA

    gmax_GABAA = initW * gmax

    record_use = u0
    record_Pr = u0
}


BREAKPOINT {
    SOLVE state METHOD cnexp
    g = gmax_GABAA*(B_GABAA-A_GABAA) :compute time varying conductance as the difference of state variables B_GABAA and A_GABAA
    i = g*(v-e_GABAA) :compute the GABAA driving force based on the time varying conductance, membrane potential, and GABAA reversal
}


DERIVATIVE state{
    A_GABAA' = -A_GABAA/tau_r_GABAA
    B_GABAA' = -B_GABAA/tau_d_GABAA
}


NET_RECEIVE (weight, R, u, tsyn (ms)){
    LOCAL Pr, weight_GABAA
    INITIAL{
        R=1
        u=u0
        tsyn=t
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
        u = u*exp(-(t - tsyn)/Fac) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
        u = u + Use*(1-u) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.
    } else {
        u = Use
    }

    if (Dep > 0) {
        R  = 1 - (1-R) * exp(-(t-tsyn)/Dep) :Probability R for a vesicle to be available for release, analogous to the pool of synaptic
                                        :resources available for release in the deterministic model. Eq. 3 in Fuhrmann et al.
        Pr = u * R                      :Pr is calculated as R * u (running value of Use)
        R  = R - u * R                  :update R as per Eq. 3 in Fuhrmann et al.
    } else {
        Pr = u 
    }

    record_use = u
    record_Pr = Pr

    if( verboseLevel > 0 ) {
        printf("Synapse %f at time %g: R = %g Pr = %g\n", synapseID, t, R, Pr )
    }

    tsyn = t

    weight_GABAA = Pr*weight*factor_GABAA
    A_GABAA = A_GABAA + weight_GABAA
    B_GABAA = B_GABAA + weight_GABAA

    if( verboseLevel > 0 ) {
        printf( " vals %g %g %g %g\n", A_GABAA, weight_GABAA, factor_GABAA, weight )
    }

    if (flag == 1) {
        : self event to set next weight at delay
          weight = conductance
    }
    : flag == 0, i.e. a spike has arrived
}


FUNCTION toggleVerbose() {
    verboseLevel = 1-verboseLevel
}
