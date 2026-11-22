import pandas as pd, numpy as np, urllib.request, os
cols=['game_id','season','season_type','posteam','defteam','home_team','away_team','home_score','away_score','down','play_type','yards_gained','success','epa','sack','qb_hit','qb_dropback','interception','fumble_lost']
frames=[]
for y in range(2010,2027):
    f=f'pbp{y}.parquet'
    try:
        if not os.path.exists(f):
            urllib.request.urlretrieve(f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet',f)
        d=pd.read_parquet(f,columns=cols); frames.append(d[d.season_type=='REG']); print(y,len(d),flush=True)
    except Exception as e: print(y,'ERR',e)
p=pd.concat(frames)
g=p.drop_duplicates('game_id')[['game_id','season','home_team','away_team','home_score','away_score']].dropna(subset=['home_score'])
rows=[]
for _,r in g.iterrows():
    hw=1.0 if r.home_score>r.away_score else 0.5 if r.home_score==r.away_score else 0.0
    rows+= [(r.season,r.home_team,hw),(r.season,r.away_team,1-hw)]
w=pd.DataFrame(rows,columns=['season','team','w']).groupby(['season','team']).agg(win_pct=('w','mean'),games=('w','size')).reset_index()
o=p[p.posteam.notna()]
fd=o[(o.down==1)&o.play_type.isin(['pass','run'])].groupby(['season','posteam']).agg(fd_ypp=('yards_gained','mean'),fd_success=('success','mean'),fd_epa=('epa','mean')).reset_index().rename(columns={'posteam':'team'})
db=o[o.qb_dropback==1].groupby(['season','posteam']).agg(sack_rate=('sack','mean'),hit_rate=('qb_hit','mean')).reset_index().rename(columns={'posteam':'team'})
db['olpass']=-(db.sack_rate+db.hit_rate)
give=o.groupby(['season','posteam']).agg(g=('interception','sum'),f=('fumble_lost','sum')).reset_index(); give['give']=give.g+give.f
t=p[p.defteam.notna()].groupby(['season','defteam']).agg(ti=('interception','sum'),tf=('fumble_lost','sum')).reset_index(); t['take']=t.ti+t.tf
to=give.rename(columns={'posteam':'team'})[['season','team','give']].merge(t.rename(columns={'defteam':'team'})[['season','team','take']])
to['to_margin']=to['take']-to['give']
df=w.merge(fd).merge(db).merge(to)
df['to_pg']=df.to_margin/df.games
df.to_csv('team_seasons.csv',index=False)
full=df[df.season<=2025]
print('N all',len(df),'N full',len(full), 'max season games', df[df.season==2026].games.max())
for name,d in [('2010-2025',full),('2010-2026',df)]:
    print(name)
    for c in ['olpass','sack_rate','hit_rate','fd_ypp','fd_success','fd_epa','to_pg']:
        print(' ',c,round(d[c].corr(d.win_pct),3))
X=full[['olpass','fd_ypp','to_pg']]; Z=(X-X.mean())/X.std(); y=(full.win_pct-full.win_pct.mean())/full.win_pct.std()
A=np.c_[np.ones(len(Z)),Z.values]; b,res,_,_=np.linalg.lstsq(A,y.values,rcond=None)
pred=A@b; r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
print('std betas',dict(zip(X.columns,b[1:].round(3))),'R2',round(r2,3))
