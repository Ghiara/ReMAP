
from third_party.CARE.bnpy.obsmodel.DiagGaussObsModel import DiagGaussObsModel
from third_party.CARE.bnpy.obsmodel.GaussObsModel import GaussObsModel
from third_party.CARE.bnpy.obsmodel.ZeroMeanGaussObsModel import ZeroMeanGaussObsModel
from third_party.CARE.bnpy.obsmodel.AutoRegGaussObsModel import AutoRegGaussObsModel
from third_party.CARE.bnpy.obsmodel.MultObsModel import MultObsModel
from third_party.CARE.bnpy.obsmodel.BernObsModel import BernObsModel
from third_party.CARE.bnpy.obsmodel.GaussRegressYFromFixedXObsModel \
	import GaussRegressYFromFixedXObsModel
from third_party.CARE.bnpy.obsmodel.GaussRegressYFromDiagGaussXObsModel \
	import GaussRegressYFromDiagGaussXObsModel

ObsModelConstructorsByName = {
    'DiagGauss': DiagGaussObsModel,
    'Gauss': GaussObsModel,
    'ZeroMeanGauss': ZeroMeanGaussObsModel,
    'AutoRegGauss': AutoRegGaussObsModel,
    'GaussRegressYFromFixedX': GaussRegressYFromFixedXObsModel,
    'GaussRegressYFromDiagGaussX': GaussRegressYFromDiagGaussXObsModel,
    'Mult': MultObsModel,
    'Bern': BernObsModel,
}

# Make constructor accessible by nickname and fullname
# Nickname = 'Gauss'
# Fullname = 'GaussObsModel'
for val in list(ObsModelConstructorsByName.values()):
    fullname = str(val.__name__)
    ObsModelConstructorsByName[fullname] = val

ObsModelNameSet = set(ObsModelConstructorsByName.keys())
