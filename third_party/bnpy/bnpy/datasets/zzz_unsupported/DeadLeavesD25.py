import third_party.CARE.bnpy.datasets.zzz_unsupported.DeadLeaves as DL
from third_party.CARE.bnpy.datasets.zzz_unsupported.DeadLeaves import get_data, get_short_name, get_data_info

DL.makeTrueParams(25)

if __name__ == '__main__':
    DL.plotTrueCovMats(doShowNow=False)
    DL.plotImgPatchPrototypes()
