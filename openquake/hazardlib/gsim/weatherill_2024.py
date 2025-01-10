# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`Weatherill2024ESHM20AvgSA`,
               :class:`Weatherill2024ESHM20SlopeGeologyAvgSA`,
               :class:`Weatherill2024ESHM20AvgSAHomoskedastic`

"""
from openquake.hazardlib.imt import AvgSA
from openquake.hazardlib.gsim.kotha_2020 import KothaEtAl2020ESHM20
from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib import const


class Weatherill2024ESHM20AvgSA(KothaEtAl2020ESHM20):
    """
    This class implements a variation of the Kotha et al (2020; 2022) GMM that
    was used for the ESHM20, but here the predicted intensity measure is
    average SA (AvgSA) rather than SA. This is a form of direct AvgSA GMM,
    which is fit using the same data set as that of KothaEtAl2020 with AvgSA
    defined according the specifications of (among others) Iacoletti et al.
    (2023):

    AvgSA = sqrt(prod([0.2 x T <= T <= 1.5 x T]))

    where a total of 10 linearly-spaced conditioning periods in the range are
    used to define the average SA.

    As the same regression methods were used to fit AvgSA then all of the
    adjustment terms adopted by the ESHM20 (sigma_mu_epsilon, c3_epsilon,
    ergodic etc.) can be applied to the AvgSA GMM, which allows the same logic
    tree to be constructed for the direct AvgSA case.

    Further details on the compilation and application of the GMM are being
    developed in the following publication (in preparation):

    Weatherill, G (2024) "A Regionalised Direct AvgSA Ground Motion Model for
    Europe", (Journal TBC)

    As this is in preparation, futue changes to the model are possible
    so we therefore retain the experimental warning, which will be removed
    at a future date.
    """
    experimental = True

    #: Set of :mod:`intensity measure types <openquake.hazardlib.imt>`
    #: this GSIM can calculate. A set should contain classes from module
    #: :mod:`openquake.hazardlib.imt`.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {AvgSA, }

    #: Supported standard deviation types is are only total std.dev
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL,
                                            const.StdDev.INTER_EVENT,
                                            const.StdDev.INTRA_EVENT}

    #: Required site parameters are vs30, vs30measured and the eshm20_region
    REQUIRES_SITES_PARAMETERS = set(("region", "vs30", "vs30measured"))

    kind = "avgsa_ESHM20"

    # Coefficients obtained direclty from the regression outputs
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt                       e1              b1              b2              b3               c1              c2               c3          tau_c3          phis2s     tau_event_0         tau_l2l           phi_0       d0_obs       d1_obs   phi_s2s_obs       d0_inf       d1_inf   phi_s2s_inf
    AvgSA(0.050)    4.3468615401    2.1512063226    0.3717451438    0.3840026031    -1.5844962290    0.3056929868    -0.6358197233    0.2763393183    0.6594454507    0.4280092597    0.5380302118    0.4676597188   2.61783357  -0.41984752    0.40314318   1.88712924  -0.29623828    0.54897643
    AvgSA(0.100)    4.6536804102    2.1243118148    0.3569931311    0.3628813315    -1.5186700529    0.2841907202    -0.7361755939    0.3101014049    0.6722188832    0.4398426213    0.5568916029    0.4648998708   2.63178976  -0.42260122    0.42319496   1.78617139  -0.28015514    0.57163410
    AvgSA(0.150)    4.7945771137    2.1274812721    0.3288504315    0.3795627756    -1.4389896655    0.2501908703    -0.7667357492    0.3157034067    0.6633675533    0.4434411516    0.5202488389    0.4582624492   2.71796680  -0.43603910    0.43472118   1.78935693  -0.28055203    0.57516812
    AvgSA(0.200)    4.8268358912    2.1435342480    0.3057719029    0.4094215327    -1.3832602354    0.2227051661    -0.7550607691    0.3110395382    0.6458262629    0.4435212138    0.4841042855    0.4498230477   2.86217015  -0.45823486    0.43749194   1.87918582  -0.29464359    0.56498414
    AvgSA(0.250)    4.8051268888    2.1754139829    0.2918562846    0.4426226923    -1.3412962888    0.2019136688    -0.7305447434    0.3035029231    0.6280883485    0.4407924746    0.4529410313    0.4425130460   3.06323423  -0.48950776    0.43044807   2.04792489  -0.32117497    0.54608798
    AvgSA(0.300)    4.7620900260    2.2141154082    0.2835145315    0.4665423314    -1.3070686825    0.1860974192    -0.7071940818    0.2957328792    0.6153905644    0.4369210190    0.4322453716    0.4352045167   3.26556277  -0.52103527    0.41182470   2.25496936  -0.35374018    0.52720419
    AvgSA(0.400)    4.6332719548    2.2947237222    0.2741828304    0.5159131190    -1.2627673311    0.1619710217    -0.6499258739    0.2794709669    0.5989732123    0.4255359229    0.3951049711    0.4216267430   3.49875383  -0.55740129    0.39503497   2.47642829  -0.38853586    0.50678238
    AvgSA(0.500)    4.4778294883    2.3662343667    0.2696192831    0.5660951786    -1.2319346319    0.1446120370    -0.5980583921    0.2666824382    0.5902397100    0.4169401398    0.3687464707    0.4117459145   3.71388275  -0.59068507    0.37900446   2.68871629  -0.42186426    0.49004842
    AvgSA(0.600)    4.3185533698    2.4277720222    0.2652148003    0.6175414034    -1.2077156215    0.1318085467    -0.5511719494    0.2536315305    0.5876115938    0.4103320095    0.3515066109    0.4034543724   3.87941606  -0.61580447    0.36505069   2.86525837  -0.44955313    0.47419374
    AvgSA(0.700)    4.1672194228    2.4894825880    0.2650749306    0.6679669266    -1.1874512590    0.1228799964    -0.5133671436    0.2433881796    0.5867287159    0.4065674107    0.3347690807    0.3963077735   3.97630461  -0.62992274    0.36262323   2.98664886  -0.46856927    0.46461818
    AvgSA(0.800)    4.0288646652    2.5542391769    0.2695180409    0.7085251974    -1.1727356202    0.1173776862    -0.4805689498    0.2315398128    0.5861237737    0.4053007828    0.3200552353    0.3900808395   4.04577748  -0.63992675    0.36701024   3.08292173  -0.48365672    0.45841564
    AvgSA(0.900)    3.8947701849    2.6046828029    0.2708607782    0.7424403141    -1.1636251550    0.1140958019    -0.4495322496    0.2191350303    0.5868406504    0.4054880421    0.3128836707    0.3847701262   4.09468998  -0.64675747    0.37430943   3.17474995  -0.49805311    0.45633438
    AvgSA(1.000)    3.7694600868    2.6564485947    0.2755580747    0.7773839055    -1.1569230243    0.1119008735    -0.4228857116    0.2077140636    0.5883712385    0.4076373926    0.3052586715    0.3809345790   4.11952534  -0.64980249    0.38194385   3.26430061  -0.51212861    0.45553548
    AvgSA(1.250)    3.4706631778    2.7640114111    0.2863971445    0.8515842010    -1.1500139705    0.1098964051    -0.3642458982    0.1878359113    0.5911312275    0.4175841723    0.2877202407    0.3732270011   4.11297148  -0.64828200    0.38649790   3.37305996  -0.52913277    0.45322510
    AvgSA(1.500)    3.2190113497    2.8630381148    0.3011679747    0.8833932182    -1.1522344562    0.1111509093    -0.3154533960    0.1759946561    0.5916370848    0.4251689388    0.2791019570    0.3684836574   4.09477274  -0.64520087    0.38407000   3.48254160  -0.54618847    0.45284527
    AvgSA(1.750)    3.0052586300    2.9878647999    0.3413194350    0.9057249259    -1.1533431170    0.1156390944    -0.2832581008    0.1664335521    0.5921743466    0.4297625190    0.2848615012    0.3646174735   4.04481293  -0.63766481    0.38178972   3.54860458  -0.55647158    0.45407701
    AvgSA(2.000)    2.7977898222    3.0610742969    0.3563412955    0.9382679687    -1.1527918383    0.1203796346    -0.2618124393    0.1609297797    0.5915803874    0.4379178489    0.2782289105    0.3621184180   3.98022776  -0.62800105    0.37429046   3.58857586  -0.56254997    0.45902122
    AvgSA(2.500)    2.4526046480    3.2447441319    0.4175895089    0.9935574572    -1.1555045836    0.1319754237    -0.2362881913    0.1606151346    0.5851395458    0.4487655553    0.2727004457    0.3564559472   3.88094598  -0.61300917    0.36664751   3.58945419  -0.56256504    0.46109358
    AvgSA(3.000)    2.1846343267    3.3795075185    0.4680279735    1.0273202155    -1.1533536786    0.1423911641    -0.2329857032    0.1478724120    0.5834092888    0.4569015623    0.2736582001    0.3508293944   3.78141847  -0.59784839    0.35646268   3.51803344  -0.55144112    0.45739762
    AvgSA(3.500)    1.9409098870    3.4656531050    0.5118696601    1.0856421414    -1.1701293398    0.1553162861    -0.2020706851    0.1410898528    0.5750292172    0.4425943058    0.2958590722    0.3492035567   3.60781997  -0.57117670    0.34930242   3.48405639  -0.54602924    0.45663788
    AvgSA(4.000)    1.7015054228    3.5163002844    0.5312213268    1.1456993040    -1.1841200080    0.1711021029    -0.1838747563    0.1437446953    0.5669140794    0.4500152389    0.2825370206    0.3499904629   3.43147290  -0.54394985    0.34725671   3.44930328  -0.54050316    0.45782846
    AvgSA(4.500)    1.6213774162    3.7391842699    0.6641371123    1.1288485321    -1.1829236019    0.1847235866    -0.1807585867    0.1213816721    0.5757075561    0.4340712588    0.2587634879    0.3381428158   3.24877975  -0.51560770    0.35330607   3.43813135  -0.53856720    0.46219582
    AvgSA(5.000)    1.4308131096    3.7717226445    0.6801268753    1.1816942442    -1.1999994884    0.1988877433    -0.1645299267    0.1239688756    0.5681631639    0.4376824137    0.2528453711    0.3391870321   3.07021475  -0.48775155    0.37025310   3.46440379  -0.54233758    0.47057169
    """)


class Weatherill2024ESHM20SlopeGeologyAvgSA(Weatherill2024ESHM20AvgSA):
    """
    Adaptation of the ESHM20-implemented Kotha et al. (2020) model taking
    direct Average Sa (AvgSA). For use when defining site amplification
    based on with slope and geology rather than inferred/measured Vs30.
    """
    experimental = True

    kind = "avgsa_ESHM20_geology"

    #: Set of :mod:`intensity measure types <openquake.hazardlib.imt>`
    #: this GSIM can calculate. A set should contain classes from module
    #: :mod:`openquake.hazardlib.imt`.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {AvgSA, }

    #: Supported standard deviation types is are only total std.dev
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL,
                                            const.StdDev.INTER_EVENT,
                                            const.StdDev.INTRA_EVENT}

    #: Required site parameter is not set
    REQUIRES_SITES_PARAMETERS = {"region", "slope", "geology"}

    #: Geological Units
    GEOLOGICAL_UNITS = [b"CENOZOIC", b"HOLOCENE", b"JURASSIC-TRIASSIC",
                        b"CRETACEOUS", b"PALEOZOIC", b"PLEISTOCENE",
                        b"PRECAMBRIAN", b"UNKNOWN"]

    COEFFS_FIXED = CoeffsTable(sa_damping=5, table="""\
    imt                      V1            V2       phi_s2s
    AvgSA(0.0500)   -0.23505110   -0.09359789    0.57524777
    AvgSA(0.1000)   -0.22027431   -0.08881495    0.58812515
    AvgSA(0.1500)   -0.21840469   -0.08830691    0.58411598
    AvgSA(0.2000)   -0.22694233   -0.09108064    0.56863949
    AvgSA(0.2500)   -0.24488926   -0.09681041    0.54758380
    AvgSA(0.3000)   -0.26548340   -0.10244796    0.52611308
    AvgSA(0.4000)   -0.28638555   -0.10745711    0.50708288
    AvgSA(0.5000)   -0.30532380   -0.11082860    0.49211675
    AvgSA(0.6000)   -0.32013378   -0.11294452    0.48141461
    AvgSA(0.7000)   -0.32893266   -0.11272525    0.47630018
    AvgSA(0.8000)   -0.33648250   -0.11327263    0.47333633
    AvgSA(0.9000)   -0.34306372   -0.11348846    0.47079169
    AvgSA(1.0000)   -0.35113716   -0.11364569    0.46851346
    AvgSA(1.2500)   -0.36007402   -0.11466989    0.46559047
    AvgSA(1.5000)   -0.37022391   -0.11702849    0.46276343
    AvgSA(1.7500)   -0.37390828   -0.11724920    0.45873202
    AvgSA(2.0000)   -0.37410840   -0.11642190    0.45449554
    AvgSA(2.5000)   -0.36880152   -0.11260637    0.45001503
    AvgSA(3.0000)   -0.35812342   -0.10696009    0.44400554
    AvgSA(3.5000)   -0.35244242   -0.10332236    0.44052465
    AvgSA(4.0000)   -0.35057632   -0.10238648    0.43874188
    AvgSA(4.5000)   -0.35153004   -0.10279447    0.43845098
    AvgSA(5.0000)   -0.35501872   -0.10406843    0.43952854
    """)

    COEFFS_RANDOM_INT = CoeffsTable(sa_damping=5, table="""\
    imt              PRECAMBRIAN      PALEOZOIC   JURASSIC-TRIASSIC     CRETACEOUS       CENOZOIC    PLEISTOCENE      HOLOCENE        UNKNOWN
    AvgSA(0.0500)     0.04440431    -0.02892322         -0.13716881    -0.05230186    -0.12125642     0.13765416    0.06883992     0.08875193
    AvgSA(0.1000)     0.05826280    -0.03109745         -0.13813473    -0.06352593    -0.13533973     0.15914360    0.05630794     0.09438349
    AvgSA(0.1500)     0.06136444    -0.03582568         -0.14426874    -0.07207127    -0.13073530     0.16589340    0.05878894     0.09685420
    AvgSA(0.2000)     0.05489489    -0.04393126         -0.15284732    -0.07872104    -0.11334863     0.16353078    0.07350189     0.09692070
    AvgSA(0.2500)     0.03889834    -0.05700330         -0.16243929    -0.08626987    -0.08802799     0.15550703    0.10516672     0.09416837
    AvgSA(0.3000)     0.01726904    -0.07875437         -0.16472931    -0.08869378    -0.06875478     0.16247269    0.12386386     0.09732666
    AvgSA(0.4000)    -0.00698559    -0.09852521         -0.16288454    -0.09293336    -0.04994895     0.17192187    0.14216443     0.09719135
    AvgSA(0.5000)    -0.03405274    -0.10755560         -0.15122228    -0.09344633    -0.02605248     0.17131167    0.15637188     0.08464589
    AvgSA(0.6000)    -0.05530501    -0.11151177         -0.14050049    -0.09475509    -0.00910426     0.17319342    0.16415751     0.07382569
    AvgSA(0.7000)    -0.07046497    -0.10186102         -0.12503738    -0.09121494     0.00954194     0.15915530    0.16881371     0.05106735
    AvgSA(0.8000)    -0.08100857    -0.10108175         -0.11704089    -0.09029657     0.01785638     0.15923935    0.17297038     0.03936166
    AvgSA(0.9000)    -0.08690111    -0.10550544         -0.10866281    -0.08778019     0.01591340     0.17214756    0.16417060     0.03661799
    AvgSA(1.0000)    -0.09354606    -0.10486674         -0.09964153    -0.08371551     0.01831340     0.17557179    0.16050531     0.02737934
    AvgSA(1.2500)    -0.09940881    -0.10111492         -0.09054299    -0.07908341     0.02037411     0.17235513    0.15624033     0.02118056
    AvgSA(1.5000)    -0.10716225    -0.09805908         -0.08525698    -0.07701466     0.02482292     0.16497320    0.16166599     0.01603086
    AvgSA(1.7500)    -0.11240712    -0.09640150         -0.07907923    -0.07492911     0.02475436     0.16353442    0.15582629     0.01870188
    AvgSA(2.0000)    -0.11302015    -0.09738500         -0.07435901    -0.07247754     0.01973015     0.17174738    0.14805774     0.01770642
    AvgSA(2.5000)    -0.10428439    -0.08655460         -0.06468280    -0.06571842     0.01668539     0.16211274    0.13643014     0.00601195
    AvgSA(3.0000)    -0.09353572    -0.07076234         -0.05435509    -0.05966450     0.01505710     0.14951429    0.12893961    -0.01519336
    AvgSA(3.5000)    -0.08535614    -0.05972990         -0.04931505    -0.05545398     0.01676627     0.13843245    0.12889559    -0.03423925
    AvgSA(4.0000)    -0.08408289    -0.05771713         -0.05213470    -0.05701903     0.02138313     0.13457367    0.14136352    -0.04636655
    AvgSA(4.5000)    -0.08523161    -0.06157877         -0.05678981    -0.05891177     0.02493221     0.13809193    0.14938582    -0.04989800
    AvgSA(5.0000)    -0.08707645    -0.07066086         -0.06023941    -0.05852453     0.02485745     0.15039280    0.14328560    -0.04203460
    """)

    COEFFS_RANDOM_GRAD = CoeffsTable(sa_damping=5, table="""\
    imt              PRECAMBRIAN      PALEOZOIC   JURASSIC-TRIASSIC    CRETACEOUS       CENOZOIC     PLEISTOCENE       HOLOCENE       UNKNOWN
    AvgSA(0.0500)     0.01019174    -0.00671749         -0.03159087   -0.01206327    -0.02807261      0.03183337     0.01588836    0.02053076
    AvgSA(0.1000)     0.01425687    -0.00742289         -0.03359498   -0.01538489    -0.03264778      0.03842344     0.01358267    0.02278758
    AvgSA(0.1500)     0.01437239    -0.00841885         -0.03389235   -0.01686420    -0.03065271      0.03889881     0.01385255    0.02270436
    AvgSA(0.2000)     0.01170863    -0.00949656         -0.03253001   -0.01682497    -0.02449309      0.03518655     0.01560313    0.02084633
    AvgSA(0.2500)     0.00705422    -0.01006083         -0.03005226   -0.01631633    -0.01598760      0.02857851     0.01964204    0.01714224
    AvgSA(0.3000)     0.00301254    -0.01247711         -0.02445906   -0.01294305    -0.01138575      0.02565267     0.01718870    0.01541106
    AvgSA(0.4000)     0.00060871    -0.01269964         -0.01810166   -0.00958284    -0.00847633      0.02227698     0.01287763    0.01309715
    AvgSA(0.5000)    -0.00071633    -0.00917333         -0.00896442   -0.00463103    -0.00476396      0.01421873     0.00609513    0.00793522
    AvgSA(0.6000)     0.00058715    -0.00464656         -0.00228301   -0.00076376    -0.00363741      0.00707727    -0.00090492    0.00457125
    AvgSA(0.7000)     0.00300945     0.00312199          0.00490062    0.00364847    -0.00123591     -0.00482704    -0.00761969   -0.00099789
    AvgSA(0.8000)     0.00494318     0.00665256          0.00804901    0.00624613    -0.00087300     -0.01059914    -0.01156916   -0.00284959
    AvgSA(0.9000)     0.00845968     0.00738209          0.01049663    0.00927053    -0.00373428     -0.01211807    -0.01873849   -0.00101808
    AvgSA(1.0000)     0.01281811     0.01009466          0.01266798    0.01167071    -0.00547594     -0.01734720    -0.02466090    0.00023260
    AvgSA(1.2500)     0.01804985     0.01425655          0.01528740    0.01433855    -0.00642659     -0.02469494    -0.03057985   -0.00023097
    AvgSA(1.5000)     0.02178052     0.01862543          0.01683443    0.01585680    -0.00566692     -0.03235509    -0.03269212   -0.00238303
    AvgSA(1.7500)     0.02602045     0.02302291          0.01921894    0.01736831    -0.00599799     -0.03694157    -0.03760119   -0.00508986
    AvgSA(2.0000)     0.02903551     0.02303283          0.01974201    0.01849300    -0.00773962     -0.03724914    -0.04329328   -0.00202131
    AvgSA(2.5000)     0.03335929     0.02443580          0.02115975    0.02082427    -0.00890809     -0.04352827    -0.05107738    0.00373462
    AvgSA(3.0000)     0.03582721     0.02429116          0.02152150    0.02287830    -0.00905799     -0.04962435    -0.05646569    0.01062986
    AvgSA(3.5000)     0.03675706     0.02468699          0.02183835    0.02387250    -0.00893185     -0.05516898    -0.05878460    0.01573055
    AvgSA(4.0000)     0.03384701     0.02392437          0.02059362    0.02285885    -0.00748610     -0.05664430    -0.05437988    0.01728643
    AvgSA(4.5000)     0.03093938     0.02341728          0.01998863    0.02137510    -0.00724577     -0.05478948    -0.05063722    0.01695209
    AvgSA(5.0000)     0.03004586     0.02383842          0.02117566    0.02018591    -0.00963392     -0.04934248    -0.05148373    0.01521428
    """)


class Weatherill2024ESHM20AvgSAHomoskedastic(Weatherill2024ESHM20AvgSA):
    """Variant of the Weatherill2024ESHM20 direct GMPE for AvgSA with the
    homoskedastic sigma coming from the original mixed effects regression
    """
    experimental = True

    kind = "avgsa_ESHM20_homoskedastic"
