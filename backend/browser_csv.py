"""Browser CSV schema; preserve raw events, aggregate equal timestamps for analysis."""
import io
import numpy as np
import pandas as pd
ACC=['acc_x_m_s2','acc_y_m_s2','acc_z_m_s2']
GYR=['gyro_x_rad_s','gyro_y_rad_s','gyro_z_rad_s']
COLS=['seconds_elapsed','acc_source',*ACC,*GYR]
def is_browser_csv(content):
    return set(COLS).issubset(pd.read_csv(io.BytesIO(content),nrows=0).columns)
def read_browser(content):
    df=pd.read_csv(io.BytesIO(content),usecols=COLS)
    for c in ['seconds_elapsed',*ACC,*GYR]:
        df[c]=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    t=df.seconds_elapsed.dropna(); span=float(t.max()-t.min()) if len(t) else 0
    quality={'events':len(df),'duplicate_timestamps':int(t.duplicated().sum()),'event_hz':round((len(t)-1)/span,2) if span>0 else None,'unique_hz':round((t.nunique()-1)/span,2) if span>0 else None,'linear_events':int((df.acc_source=='linear').sum()),'gyro_valid':int(df[GYR].notna().all(axis=1).sum()),'acc_valid':int(df[ACC].notna().all(axis=1).sum()),'duration_seconds':round(span,3)}
    data={}
    for name,cols,mask in [('Accelerometer',ACC,df.acc_source=='linear'),('TotalAcceleration',ACC,df.acc_source=='with_gravity'),('Gyroscope',GYR,pd.Series(True,index=df.index))]:
        sub=df.loc[mask,['seconds_elapsed',*cols]].dropna(); sub=sub[sub.seconds_elapsed>=0]
        if len(sub): data[name]=sub.groupby('seconds_elapsed',as_index=False).mean().rename(columns=dict(zip(cols,['x','y','z'])))
    return data,quality
