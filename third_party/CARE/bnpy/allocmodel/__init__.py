from third_party.CARE.bnpy.allocmodel.AllocModel import AllocModel

from third_party.CARE.bnpy.allocmodel.mix.FiniteMixtureModel import FiniteMixtureModel
from third_party.CARE.bnpy.allocmodel.mix.DPMixtureModel import DPMixtureModel
from third_party.CARE.bnpy.allocmodel.mix.DPMixtureRestrictedLocalStep import make_xPiVec_and_emptyPi

from third_party.CARE.bnpy.allocmodel.topics.FiniteTopicModel import FiniteTopicModel
from third_party.CARE.bnpy.allocmodel.topics.HDPTopicModel import HDPTopicModel

from third_party.CARE.bnpy.allocmodel.hmm.FiniteHMM import FiniteHMM
from third_party.CARE.bnpy.allocmodel.hmm.HDPHMM import HDPHMM

from third_party.CARE.bnpy.allocmodel.relational.FiniteSMSB import FiniteSMSB
from third_party.CARE.bnpy.allocmodel.relational.FiniteMMSB import FiniteMMSB
from third_party.CARE.bnpy.allocmodel.relational.FiniteAssortativeMMSB import FiniteAssortativeMMSB
from third_party.CARE.bnpy.allocmodel.relational.HDPMMSB import HDPMMSB
from third_party.CARE.bnpy.allocmodel.relational.HDPAssortativeMMSB import HDPAssortativeMMSB


AllocModelConstructorsByName = {
    'FiniteMixtureModel': FiniteMixtureModel,
    'DPMixtureModel': DPMixtureModel,
    'FiniteTopicModel': FiniteTopicModel,
    'HDPTopicModel': HDPTopicModel,
    'FiniteHMM': FiniteHMM,
    'HDPHMM': HDPHMM,
    'FiniteSMSB': FiniteSMSB,
    'FiniteMMSB': FiniteMMSB,
    'FiniteAssortativeMMSB': FiniteAssortativeMMSB,
    'HDPMMSB': HDPMMSB,
    'HDPAssortativeMMSB': HDPAssortativeMMSB,
}

AllocModelNameSet = set(AllocModelConstructorsByName.keys())

__all__ = ['AllocModel']
for name in AllocModelConstructorsByName:
    __all__.append(name)
