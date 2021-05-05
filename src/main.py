import pandas as pd
import larch
from larch.roles import PX, P, X
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname", help="file name", type=Path)
    args = parser.parse_args()

num_cols = ['OPERAT_FREQ_CNT', "CIRCUITY_RATIO", 'OD_ELAPSED_TIME_RATIO',
            "LEG_CDSHR_BREADTH_RATIO", "LEG_CDSHR_DEPTH_RATIO", "OD_BEST_SRVC_RATIO", "OD_BEST_SRVC_TM_DIFF_RATIO"]
cat_cols = ["OD_OPERAT_AIRLN_TYPE_CD"]
svc_cols = ['OD_SRVC_TYPE_CD', 'OD_STOP_OVER_QTY', 'OD_CONNECT_QTY']
larch_cols = ['ID_CASE', 'ID_ALT', 'CHOICE', 'POO_pax']
tod_cols = ["CUR_POO", "OD_LOCAL_OUT_HR", "OD_TM_OFFSET_HR"]

df = pd.read_parquet("data/sample_jul_input.parquet", columns=num_cols + cat_cols + larch_cols + svc_cols + tod_cols)
df.set_index(['ID_CASE', 'ID_ALT'], inplace=True, drop=False)

df_larch = larch.DataFrames(ca=df[['CHOICE'] + num_cols + cat_cols + svc_cols + tod_cols], av=True, ch='CHOICE',crack=True, autoscale_weights=True, wt="POO_pax")

# d1 = df_larch.new_systematic_alternatives(
#     groupby='OD_STOP_OVER_QTY',
#     name='alternative_code',
#     padding_levels=4,
#     groupby_prefixes=['STOP'],
#     overwrite=False,
#     complete_features_list={'OD_STOP_OVER_QTY':[0,1,2]},
# )
#
# print(d1.info())

m = larch.Model(dataservice=df_larch)
m.title = "Market Share Choice Model using Larch"

v = num_cols
for i in df.OD_OPERAT_AIRLN_TYPE_CD.unique():
    v.append(f"OD_OPERAT_AIRLN_TYPE_CD=='{i}'")

for OD_CONNECT_QTY in df.OD_CONNECT_QTY.unique():
    for OD_STOP_OVER_QTY in df.OD_STOP_OVER_QTY.unique():
        for OD_SRVC_TYPE_CD in df.OD_SRVC_TYPE_CD.unique():
            v.append(
                f"(OD_CONNECT_QTY=={OD_CONNECT_QTY}) & (OD_STOP_OVER_QTY=={OD_STOP_OVER_QTY}) & (OD_SRVC_TYPE_CD=='{OD_SRVC_TYPE_CD}')")

for CUR_POO in df.CUR_POO.unique():
    for OD_TM_OFFSET_HR in df.OD_TM_OFFSET_HR.unique():
        for OD_LOCAL_OUT_HR in df.OD_LOCAL_OUT_HR.unique():
            v.append(
                f"(CUR_POO=='{CUR_POO}') & (OD_TM_OFFSET_HR=={OD_TM_OFFSET_HR}) & (OD_LOCAL_OUT_HR=={OD_LOCAL_OUT_HR})")

m.utility_ca = sum(PX(i) for i in v)

m.choice_ca_var = 'CHOICE'

m.ordering = [
    ("OD_OPERAT_AIRLN_TYPE_CD", 'OD_OPERAT_AIRLN_TYPE_CD.*',),
    ("SVC_TYPE_CD", '.*OD_CONNECT_QTY.*',),
]

# m.magic_nesting()
m.load_data()
m.maximize_loglike()
m.calculate_parameter_covariance()
report = larch.Reporter(title=m.title)
report << '# Parameter Summary' << m.parameter_summary('xml')
report << "# Estimation Statistics" << m.estimation_statistics()
report << "# Utility Functions" << m.utility_functions()
report.save(
    'data/market_share_choice.html',
    overwrite=True,
    metadata=m,
)
