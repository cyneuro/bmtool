COMMENT
/**
 * @file DetAMPANMDA.mod
 * @brief Adapted from ProbAMPANMDA_EMS.mod by Eilif, Michael and Srikanth
 * @author chindemi
 * @date 2014-05-25
 * @remark Copyright (c) BBP/EPFL 2005-2021. This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
* Several changes have been made from the orginal version of this synapse by Greg Glickert to better adapt the model for Large Scale BMTk/Neuron models
 */
ENDCOMMENT


TITLE AMPA and NMDA receptor with presynaptic short-term plasticity


COMMENT
AMPA and NMDA receptor conductance using a dual-exponential profile
presynaptic short-term plasticity based on Fuhrmann et al. 2002, deterministic
version.
ENDCOMMENT


NEURON {
    THREADSAFE

    POINT_PROCESS AMPA_NMDA_STP
    RANGE initW     : synaptic scaler for large scale networks added by Greg
    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA
    RANGE Use, Dep, Fac, u0, mg, NMDA_ratio
    RANGE i, i_AMPA, i_NMDA, g_AMPA, g_NMDA, g, e
    RANGE gmax_AMPA, gmax_NMDA
    NONSPECIFIC_CURRENT i
    RANGE synapseID, verboseLevel
    RANGE conductance
    GLOBAL nc_type_param
    : For debugging
    :RANGE sgid, tgid
    RANGE record_use, record_Pr
}


PARAMETER {
    initW      = 1.0         : added by Greg Glickert to scale synaptic weight for large scale modeling
    tau_r_AMPA = 0.2   (ms)  : Dual-exponential conductance profile
    tau_d_AMPA = 1.7   (ms)  : IMPORTANT: tau_r < tau_d
    tau_r_NMDA = 0.29  (ms)  : Dual-exponential conductance profile
    tau_d_NMDA = 43    (ms)  : IMPORTANT: tau_r < tau_d
    Use = 1.0          (1)   : Utilization of synaptic efficacy
    Dep = 100          (ms)  : Relaxation time constant from depression
    Fac = 10           (ms)  : Relaxation time constant from facilitation
    e = 0              (mV)  : AMPA and NMDA reversal potential
    mg = 1             (mM)  : Initial concentration of mg2+
    gmax_NMDA = .001   (uS)  : Weight conversion factor (from nS to uS)
    gmax_AMPA = .001
    u0 = 0                   : Initial value of u, which is the running value of Use
    NMDA_ratio = 0.71  (1)   : The ratio of NMDA to AMPA
    synapseID = 0
    verboseLevel = 0
    conductance = 0.0
    nc_type_param = 7
    :sgid = -1
    :tgid = -1
}


ASSIGNED {
    v (mV)
    i (nA)
    i_AMPA (nA)
    i_NMDA (nA)
    g_AMPA (uS)
    g_NMDA (uS)
    g (uS)
    factor_AMPA
    factor_NMDA
    mggate
    record_use
    record_Pr
}


STATE {
    A_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_r_AMPA
    B_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_d_AMPA
    A_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_r_NMDA
    B_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_d_NMDA
}


INITIAL{
    LOCAL tp_AMPA, tp_NMDA

    A_AMPA = 0
    B_AMPA = 0

    A_NMDA = 0
    B_NMDA = 0

    tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA) :time to peak of the conductance
    tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA) :time to peak of the conductance

    factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA) :AMPA Normalization factor - so that when t = tp_AMPA, gsyn = gpeak
    factor_AMPA = 1/factor_AMPA

    factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA) :NMDA Normalization factor - so that when t = tp_NMDA, gsyn = gpeak
    factor_NMDA = 1/factor_NMDA

    record_use = u0
    record_Pr = u0
}


BREAKPOINT {
    SOLVE state METHOD cnexp
    mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 3.57 (mM))) :mggate kinetics - Jahr & Stevens 1990
    g_AMPA = gmax_AMPA*(B_AMPA-A_AMPA) :compute time varying conductance as the difference of state variables B_AMPA and A_AMPA
    g_NMDA = gmax_NMDA*(B_NMDA-A_NMDA) * mggate :compute time varying conductance as the difference of state variables B_NMDA and A_NMDA and mggate kinetics
    g = g_AMPA + g_NMDA
    i_AMPA = g_AMPA*(v-e) :compute the AMPA driving force based on the time varying conductance, membrane potential, and AMPA reversal
    i_NMDA = g_NMDA*(v-e) :compute the NMDA driving force based on the time varying conductance, membrane potential, and NMDA reversal
    i = (i_AMPA + i_NMDA) * initW
}


DERIVATIVE state{
    A_AMPA' = -A_AMPA/tau_r_AMPA
    B_AMPA' = -B_AMPA/tau_d_AMPA
    A_NMDA' = -A_NMDA/tau_r_NMDA
    B_NMDA' = -B_NMDA/tau_d_NMDA
}


NET_RECEIVE (weight, weight_AMPA, weight_NMDA, R, Pr, u, tsyn (ms), nc_type){
    weight_AMPA = weight
    weight_NMDA = weight * NMDA_ratio

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

    if (flag == 1) {
        : self event to set next weight at delay
          weight = conductance

    }
    : flag == 0, i.e. a spike has arrived

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

    A_AMPA = A_AMPA + Pr*weight_AMPA*factor_AMPA
    B_AMPA = B_AMPA + Pr*weight_AMPA*factor_AMPA
    A_NMDA = A_NMDA + Pr*weight_NMDA*factor_NMDA
    B_NMDA = B_NMDA + Pr*weight_NMDA*factor_NMDA

    if( verboseLevel > 0 ) {
        printf( " vals %g %g %g %g\n", A_AMPA, weight_AMPA, factor_AMPA, weight )
    }
}


FUNCTION toggleVerbose() {
    verboseLevel = 1-verboseLevel
}
