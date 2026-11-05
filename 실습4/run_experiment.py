from __future__ import annotations

import hashlib, json, math, os, platform, random, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data_for_UCI_named.csv"
OUT = ROOT / "results"
TABLES, FIGS, MODELS, HIST = [OUT / x for x in ("tables", "figures", "models", "histories")]
FEATURES = ["tau1","tau2","tau3","tau4","p2","p3","p4","g1","g2","g3","g4"]

def setup():
    for p in (OUT,TABLES,FIGS,MODELS,HIST): p.mkdir(parents=True, exist_ok=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def metrics(y, pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,average="binary",pos_label=1,zero_division=0)
    return {"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),
            "precision_unstable":p,"recall_unstable":r,"f1_unstable":f}

class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(11,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(1)

def make_opt(name, params, lr):
    if name=="SGD": return torch.optim.SGD(params,lr=lr,momentum=0)
    if name=="SGD+Momentum": return torch.optim.SGD(params,lr=lr,momentum=.9,nesterov=False)
    if name=="Adam": return torch.optim.Adam(params,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    if name=="RMSprop": return torch.optim.RMSprop(params,lr=lr,alpha=.9,momentum=0,eps=1e-8,weight_decay=0,centered=False)
    raise ValueError(name)

def predict_raw(model,X):
    model.eval()
    with torch.no_grad(): return model(torch.as_tensor(X,dtype=torch.float32)).cpu().numpy()

def prediction(model,X,kind):
    raw=predict_raw(model,X)
    if kind=="classification":
        score=1/(1+np.exp(-np.clip(raw,-50,50))); pred=(score>=.5).astype(int)
    else: score=raw; pred=(raw>0).astype(int)
    return raw,score,pred

def loss_value(model,X,y,kind):
    model.eval(); xt=torch.as_tensor(X,dtype=torch.float32); yt=torch.as_tensor(y,dtype=torch.float32)
    fn=nn.BCEWithLogitsLoss() if kind=="classification" else nn.MSELoss()
    with torch.no_grad(): return float(fn(model(xt),yt))

def init_state():
    torch.manual_seed(SEED); return {k:v.clone() for k,v in MLP().state_dict().items()}

def train_model(Xtr,ytr,Xv,yv,kind,opt_name,lr,max_epochs,batch_size=64,patience=None,
                initial=None,run_id="",count_budget=False,logs=None,purpose=""):
    if count_budget:
        counter=OUT/"goal3_fit_counter.json"
        state=json.loads(counter.read_text()) if counter.exists() else {"count":0,"calls":[]}
        if state["count"]>=12: raise RuntimeError("목표 3의 13번째 fit 호출 차단")
        state["count"]+=1; state["calls"].append({"run_id":run_id,"time":pd.Timestamp.now(tz="Asia/Seoul").isoformat()})
        counter.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    model=MLP()
    if initial is not None: model.load_state_dict(initial)
    opt=make_opt(opt_name,model.parameters(),lr)
    criterion=nn.BCEWithLogitsLoss() if kind=="classification" else nn.MSELoss()
    hist=[]; best=None; best_acc=-1.; best_loss=float("inf"); stale=0; t0=time.perf_counter()
    n=len(Xtr)
    for ep in range(1,max_epochs+1):
        model.train(); gen=torch.Generator().manual_seed(SEED+ep); order=torch.randperm(n,generator=gen)
        for s in range(0,n,batch_size):
            ix=order[s:s+batch_size]; xb=torch.as_tensor(Xtr[ix],dtype=torch.float32); yb=torch.as_tensor(ytr[ix],dtype=torch.float32)
            opt.zero_grad(set_to_none=True); loss=criterion(model(xb),yb); loss.backward(); opt.step()
        trloss=loss_value(model,Xtr,ytr,kind); vloss=loss_value(model,Xv,yv,kind)
        _,_,trp=prediction(model,Xtr,kind); _,_,vp=prediction(model,Xv,kind)
        tracc=accuracy_score((ytr>0).astype(int) if kind=="regression" else ytr.astype(int),trp)
        vacc=accuracy_score((yv>0).astype(int) if kind=="regression" else yv.astype(int),vp)
        hist.append({"epoch":ep,"train_loss":trloss,"val_loss":vloss,"train_accuracy":tracc,"val_accuracy":vacc})
        improved=(vacc>best_acc+1e-12) or (abs(vacc-best_acc)<=1e-12 and vloss<best_loss-1e-12)
        if improved: best_acc,best_loss,best,stale=vacc,vloss,{k:v.detach().clone() for k,v in model.state_dict().items()},0
        else: stale+=1
        if patience is not None and stale>=patience: break
    elapsed=time.perf_counter()-t0
    # Goal 2 is defined by epoch 50; goal 3 restores its selected checkpoint.
    if patience is not None: model.load_state_dict(best)
    h=pd.DataFrame(hist); h.to_csv(HIST/f"{run_id}.csv",index=False)
    yvc=(yv>0).astype(int) if kind=="regression" else yv.astype(int); _,_,vp=prediction(model,Xv,kind); mm=metrics(yvc,vp)
    rec={"run_id":run_id,"purpose":purpose,"model_type":kind,"optimizer":opt_name,"learning_rate":lr,"seed":SEED,
         "n_train":len(Xtr),"n_validation":len(Xv),"n_evaluation":0,"max_epoch":max_epochs,"actual_epochs":len(h),
         "best_epoch":int(h.sort_values(["val_accuracy","val_loss"],ascending=[False,True]).iloc[0].epoch),
         **{"val_"+k:v for k,v in mm.items()},"val_loss":loss_value(model,Xv,yv,kind),"training_seconds":elapsed,"status":"success","error":""}
    if logs is not None: logs.append(rec)
    return model,h,rec

def audit_and_split(df):
    yclass=(df.stabf=="unstable").astype(int).to_numpy()
    input_dups=int(df.drop(columns=["stab","stabf"]).duplicated().sum())
    cross=pd.crosstab(df.stab>0,df.stabf)
    audit={"rows":len(df),"columns":len(df.columns),"dtypes":{c:str(t) for c,t in df.dtypes.items()},
           "missing_total":int(df.isna().sum().sum()),"infinite_total":int(np.isinf(df.select_dtypes('number')).sum().sum()),
           "full_row_duplicates":int(df.duplicated().sum()),"input_feature_duplicates":input_dups,
           "stab_zero_count":int((df.stab==0).sum()),"power_balance_max_abs":float((df.p1+df.p2+df.p3+df.p4).abs().max()),
           "source_sha256":sha256(DATA),"class_counts":df.stabf.value_counts().to_dict(),"stab_summary":df.stab.describe().to_dict(),
           "stab_sign_label_crosstab":cross.to_dict()}
    (OUT/"data_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    idx=np.arange(len(df)); dev,test=train_test_split(idx,test_size=.2,stratify=yclass,random_state=42)
    tr,val=train_test_split(dev,test_size=.2,stratify=yclass[dev],random_state=42)
    pd.DataFrame({"index":idx,"split":np.where(np.isin(idx,test),"test",np.where(np.isin(idx,val),"validation","train"))}).to_csv(OUT/"split_indices.csv",index=False)
    return yclass,dev,test,tr,val,audit

def scaled(X,tr,others):
    sc=StandardScaler().fit(X[tr]); return sc,sc.transform(X[tr]).astype('float32'),[sc.transform(X[z]).astype('float32') for z in others]

def cm_save(y,p,name):
    cm=confusion_matrix(y,p,labels=[0,1]); pd.DataFrame(cm,index=["actual_stable","actual_unstable"],columns=["pred_stable","pred_unstable"]).to_csv(TABLES/f"confusion_{name}.csv")
    fig,ax=plt.subplots(figsize=(4.8,4)); im=ax.imshow(cm,cmap="Blues");
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=13)
    fnr=cm[1,0]/cm[1].sum(); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["stable","unstable"],yticklabels=["stable","unstable"],xlabel="Predicted",ylabel="Actual",title=f"{name}: unstable→stable {cm[1,0]} ({fnr:.1%})")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(FIGS/f"confusion_{name}.png",dpi=180); plt.close(fig)
    return cm

def main():
    setup(); (OUT/"goal3_fit_counter.json").write_text(json.dumps({"count":0,"calls":[]},indent=2),encoding="utf-8")
    df=pd.read_csv(DATA); ycls,dev,test,tr,val,audit=audit_and_split(df)
    X=df[FEATURES].to_numpy(); yreg=df.stab.to_numpy(dtype='float32'); ybin=ycls.astype('float32')
    # EDA
    fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].hist(df.stab,bins=40,color="#4472C4"); ax[0].axvline(0,color="red",ls="--"); ax[0].set(title="stab distribution",xlabel="stab",ylabel="count")
    vc=df.stabf.value_counts(); ax[1].bar(vc.index,vc.values,color=["#70AD47","#ED7D31"]); ax[1].set(title="Class counts",ylabel="count")
    for i,v in enumerate(vc.values): ax[1].text(i,v+80,f"{v} ({v/len(df):.1%})",ha="center")
    fig.tight_layout(); fig.savefig(FIGS/"data_distribution.png",dpi=180); plt.close(fig)
    sc,Xtr,[Xv,Xtest]=scaled(X,tr,[val,test]); joblib.dump(sc,MODELS/"goal2_scaler.joblib")
    initial=init_state(); goal2logs=[]
    reg,hr,rr=train_model(Xtr,yreg[tr],Xv,yreg[val],"regression","Adam",.001,50,initial=initial,run_id="G2_REG",logs=goal2logs,purpose="회귀-분류 공정 비교")
    clf,hc,rc=train_model(Xtr,ybin[tr],Xv,ybin[val],"classification","Adam",.001,50,initial=initial,run_id="G2_CLF",logs=goal2logs,purpose="회귀-분류 공정 비교")
    pd.DataFrame(goal2logs).to_csv(OUT/"goal2_run_log.csv",index=False)
    # selection from epoch 50 validation
    def rank_g2(model,kind,y):
        _,_,p=prediction(model,Xv,kind); return metrics(y,p)
    mr=rank_g2(reg,"regression",ycls[val]); mc=rank_g2(clf,"classification",ycls[val])
    keys=lambda m:(m["accuracy"],m["balanced_accuracy"],m["recall_unstable"])
    selected="classification" if keys(mc)>=keys(mr) else "regression"
    yselected=ybin if selected=="classification" else yreg
    # curves
    fig,ax=plt.subplots(1,2,figsize=(11,4));
    for h,n in [(hr,"Regression (MSE)"),(hc,"Classification (BCE)")]: ax[0].plot(h.epoch,h.val_loss,label=n); ax[1].plot(h.epoch,h.val_accuracy,label=n)
    ax[0].set(title="Validation loss (scales not comparable)",xlabel="epoch",ylabel="loss"); ax[1].set(title="Validation decision accuracy",xlabel="epoch",ylabel="accuracy")
    for a in ax:a.legend();a.grid(alpha=.25)
    fig.tight_layout();fig.savefig(FIGS/"goal2_learning_curves.png",dpi=180);plt.close(fig)
    # Goal 3 S01-S07
    g3=[]; candidates=[]; base=[("SGD",.01),("SGD+Momentum",.01),("Adam",.001),("RMSprop",.001)]
    for i,(op,lr) in enumerate(base,1):
        m,h,r=train_model(Xtr,yselected[tr],Xv,yselected[val],selected,op,lr,100,patience=15,run_id=f"S{i:02d}",count_budget=True,logs=g3,purpose="optimizer 대표 학습률 비교")
        candidates.append({"candidate_id":f"S{i:02d}","optimizer":op,"learning_rate":lr,**{k:r["val_"+k] for k in ["accuracy","balanced_accuracy","recall_unstable"]},"best_epoch":r["best_epoch"]})
    def ckey(c): return (-c["accuracy"],-c["balanced_accuracy"],-c["recall_unstable"],int(c["candidate_id"][1:]))
    winner=sorted(candidates,key=ckey)[0]; eta=winner["learning_rate"]
    for j,lr in enumerate([eta/10,3*eta,10*eta],5):
        op=winner["optimizer"]; m,h,r=train_model(Xtr,yselected[tr],Xv,yselected[val],selected,op,lr,100,patience=15,run_id=f"S{j:02d}",count_budget=True,logs=g3,purpose="1단계 우승 optimizer 추가 학습률")
        candidates.append({"candidate_id":f"S{j:02d}","optimizer":op,"learning_rate":lr,**{k:r["val_"+k] for k in ["accuracy","balanced_accuracy","recall_unstable"]},"best_epoch":r["best_epoch"]})
    ranked=sorted(candidates,key=ckey); pd.DataFrame(ranked).assign(rank=range(1,8)).to_csv(OUT/"candidate_ranking.csv",index=False)
    top2=ranked[:2]
    # S08-S11 same outer and inner splits
    skf=StratifiedKFold(n_splits=2,shuffle=True,random_state=42); folds=list(skf.split(dev,ybin[dev])); cvrows=[]; runno=8
    for cand in top2:
        for fold,(outer_train_pos,eval_pos) in enumerate(folds,1):
            outer_idx=dev[outer_train_pos]; eval_idx=dev[eval_pos]
            inner_train,inner_val=train_test_split(outer_idx,test_size=.2,stratify=ybin[outer_idx],random_state=42)
            fsc=StandardScaler().fit(X[inner_train]); fxtr=fsc.transform(X[inner_train]).astype('float32'); fxv=fsc.transform(X[inner_val]).astype('float32'); fxe=fsc.transform(X[eval_idx]).astype('float32')
            m,h,r=train_model(fxtr,yselected[inner_train],fxv,yselected[inner_val],selected,cand["optimizer"],cand["learning_rate"],100,patience=15,run_id=f"S{runno:02d}",count_budget=True,logs=g3,purpose=f"상위 후보 {cand['candidate_id']} 2-fold fold{fold}")
            _,_,ep=prediction(m,fxe,selected); em=metrics(ybin[eval_idx].astype(int),ep); r.update({"n_evaluation":len(eval_idx),**{"fold_"+k:v for k,v in em.items()}})
            cvrows.append({"candidate_id":cand["candidate_id"],"optimizer":cand["optimizer"],"learning_rate":cand["learning_rate"],"fold":fold,"best_epoch":r["best_epoch"],**em})
            runno+=1
    cv=pd.DataFrame(cvrows); cv.to_csv(OUT/"cross_validation_results.csv",index=False)
    agg=cv.groupby(["candidate_id","optimizer","learning_rate"]).agg({"accuracy":["mean","std"],"balanced_accuracy":["mean","std"],"recall_unstable":["mean","std"]}).reset_index()
    agg.columns=['_'.join(c).strip('_') for c in agg.columns]; agg.to_csv(OUT/"cross_validation_summary.csv",index=False)
    order=agg.sort_values(["accuracy_mean","balanced_accuracy_mean","recall_unstable_mean","candidate_id"],ascending=[False,False,False,True]); finalrow=order.iloc[0]
    final_cid=finalrow.candidate_id; finalop=finalrow.optimizer; finallr=float(finalrow.learning_rate); epochs=math.floor(float(np.median(cv.loc[cv.candidate_id==final_cid,"best_epoch"]))+.5)
    # S12 final dev training; pass a fixed dummy validation only for history API, no selection/restoration
    fsc=StandardScaler().fit(X[dev]); Xdev=fsc.transform(X[dev]).astype('float32'); Xte=fsc.transform(X[test]).astype('float32')
    final,hf,rf=train_model(Xdev,yselected[dev],Xdev[:1],yselected[dev][:1],selected,finalop,finallr,epochs,patience=None,run_id="S12",count_budget=True,logs=g3,purpose="개발용 전체 최종 재학습")
    rf["n_validation"]=0; rf["val_accuracy"]=np.nan; rf["val_balanced_accuracy"]=np.nan; rf["val_precision_unstable"]=np.nan; rf["val_recall_unstable"]=np.nan; rf["val_f1_unstable"]=np.nan; rf["val_loss"]=np.nan; rf["best_epoch"]=epochs
    pd.DataFrame(g3).to_csv(OUT/"goal3_run_log.csv",index=False); joblib.dump(fsc,MODELS/"final_scaler.joblib"); torch.save({"state_dict":final.state_dict(),"model_type":selected,"features":FEATURES,"optimizer":finalop,"learning_rate":finallr,"epochs":epochs},MODELS/"final_model.pt")
    (MODELS/"feature_columns.json").write_text(json.dumps(FEATURES,indent=2),encoding="utf-8")
    # Test only now, after all settings are fixed
    results=[]; preds={}
    for name,model,kind,xx in [("goal2_regression",reg,"regression",Xtest),("goal2_classification",clf,"classification",Xtest),("final",final,selected,Xte)]:
        raw,score,p=prediction(model,xx,kind); mm=metrics(ybin[test].astype(int),p); cm_save(ybin[test].astype(int),p,name)
        rec={"model":name,"model_type":kind,**mm}
        if kind=="regression": rec.update(mae=mean_absolute_error(yreg[test],raw),rmse=mean_squared_error(yreg[test],raw)**.5)
        results.append(rec); preds[name]=(raw,score,p)
    pd.DataFrame(results).to_csv(OUT/"test_metrics.csv",index=False)
    pp=pd.DataFrame({"row_index":test,"actual_stab":yreg[test],"actual_class":ybin[test].astype(int)})
    for n,(raw,score,p) in preds.items(): pp[f"{n}_raw"]=raw;pp[f"{n}_score"]=score;pp[f"{n}_pred_class"]=p;pp[f"{n}_correct"]=(p==ybin[test])
    pp.to_csv(OUT/"test_predictions.csv",index=False)
    # Error/boundary analysis
    pp["abs_stab"]=pp.actual_stab.abs(); bins=[0,.005,.01,.02,np.inf]; labels=["[0,.005)","[.005,.01)","[.01,.02)","[.02,inf)"]; pp["distance_bin"]=pd.cut(pp.abs_stab,bins=bins,labels=labels,right=False)
    rows=[]
    for n in ["goal2_regression","goal2_classification"]:
        e=~pp[f"{n}_correct"]
        for b,g in pp.groupby("distance_bin",observed=False): rows.append({"model":n,"distance_bin":str(b),"n":len(g),"errors":int((~g[f'{n}_correct']).sum()),"error_rate":float((~g[f'{n}_correct']).mean())})
        pd.DataFrame({"group":["correct","incorrect"],"median_abs_stab":[pp.loc[~e,"abs_stab"].median(),pp.loc[e,"abs_stab"].median()]}).to_csv(TABLES/f"{n}_abs_stab_summary.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/"boundary_error_analysis.csv",index=False)
    pp.loc[~pp.goal2_regression_correct,["row_index","actual_stab","goal2_regression_raw","actual_class","goal2_regression_pred_class","abs_stab"]].to_csv(OUT/"regression_misclassifications.csv",index=False)
    fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].scatter(pp.actual_stab,pp.goal2_regression_raw,s=8,alpha=.4); lim=[min(pp.actual_stab.min(),pp.goal2_regression_raw.min()),max(pp.actual_stab.max(),pp.goal2_regression_raw.max())];ax[0].plot(lim,lim,'r--');ax[0].axhline(0,color='gray');ax[0].axvline(0,color='gray');ax[0].set(xlabel="Actual stab",ylabel="Predicted stab",title="Regression predictions")
    er=pd.DataFrame(rows); [ax[1].plot(g.distance_bin,g.error_rate,marker='o',label=n) for n,g in er.groupby('model')];ax[1].set(xlabel="|stab| bin",ylabel="misclassification rate",title="Boundary-distance errors");ax[1].tick_params(axis='x',rotation=20);ax[1].legend();fig.tight_layout();fig.savefig(FIGS/"misclassification_analysis.png",dpi=180);plt.close(fig)
    # paired bootstrap difference classification - regression
    rng=np.random.default_rng(42); cr=pp.goal2_regression_correct.to_numpy(float); cc=pp.goal2_classification_correct.to_numpy(float); dif=[]
    for _ in range(5000):
        ix=rng.integers(0,len(pp),len(pp));dif.append(cc[ix].mean()-cr[ix].mean())
    boot={"difference_definition":"classification accuracy - regression accuracy","observed":float(cc.mean()-cr.mean()),"ci95_percentile":[float(np.quantile(dif,.025)),float(np.quantile(dif,.975))],"resamples":5000,"seed":42}
    (OUT/"paired_bootstrap.json").write_text(json.dumps(boot,indent=2),encoding="utf-8")
    # versions, summary data for report
    vers={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,"device":"CPU","deterministic_algorithms":True}
    (OUT/"environment.json").write_text(json.dumps(vers,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={"selected_problem_type":selected,"final_optimizer":finalop,"final_learning_rate":finallr,"final_epochs":epochs,"goal3_fit_calls":json.loads((OUT/'goal3_fit_counter.json').read_text())["count"],"goal2_validation":{"regression":mr,"classification":mc},"test_results":results,"bootstrap":boot,"hypotheses":{"H1":"supported" if results[1]['accuracy']>results[0]['accuracy'] else "not_supported","H2":"supported" if winner['optimizer']=='Adam' else "not_supported","H3":"pending_computation"}}
    adam={c["learning_rate"]:c["accuracy"] for c in candidates if c["optimizer"]=="Adam"}
    summary["hypotheses"]["H3"]="supported" if .001 in adam and .01 in adam and adam[.001]>adam[.01] else ("not_supported" if .001 in adam and .01 in adam else "untested")
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    # measured compute extrapolation
    total=float(pd.DataFrame(g3).training_seconds.sum()); per_epoch=(pd.DataFrame(g3).training_seconds/pd.DataFrame(g3).actual_epochs).mean(); comp={"goal3_measured_seconds":total,"mean_seconds_per_executed_epoch":float(per_epoch),"exhaustive_calls_including_final":61,"rough_exhaustive_seconds_scaled_by_call_count":total*61/12,"note":"60회(4 optimizers x 5 learning rates x 3 folds)+최종 1회. 61/12 단순 비례 추정이며 조기 종료 epoch, 표본 수, 장치가 달라 실제 시간은 크게 달라질 수 있음."}
    (OUT/"compute_cost.json").write_text(json.dumps(comp,ensure_ascii=False,indent=2),encoding="utf-8")
    write_reports(summary,audit,candidates,cv,comp)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def write_reports(s,a,cands,cv,comp):
    tm={x['model']:x for x in s['test_results']}; h=s['hypotheses']
    report=f"""# 실습 4 실행 결과 해설

## 핵심 결과

- 목표 2의 50 epoch 검증 기준으로 선택한 문제 유형: **{s['selected_problem_type']}**
- 목표 3에서 선택한 설정: **{s['final_optimizer']}, learning rate {s['final_learning_rate']:g}, {s['final_epochs']} epoch**
- 목표 3 실제 학습 호출: **{s['goal3_fit_calls']}/12회**
- 최종 테스트 정확도: **{tm['final']['accuracy']:.4f}**, balanced accuracy {tm['final']['balanced_accuracy']:.4f}, unstable F1 {tm['final']['f1_unstable']:.4f}
- 목표 2 테스트 정확도: 회귀 {tm['goal2_regression']['accuracy']:.4f}, 직접 분류 {tm['goal2_classification']['accuracy']:.4f}
- paired bootstrap 정확도 차이(분류-회귀): {s['bootstrap']['observed']:.4f}, 95% CI [{s['bootstrap']['ci95_percentile'][0]:.4f}, {s['bootstrap']['ci95_percentile'][1]:.4f}]

## 가설 회고

- H1: {h['H1']}. 직접 분류의 테스트 정확도가 회귀보다 높은지로 판정했다.
- H2: {h['H2']}. 첫 4회 대표 학습률 비교 우승 optimizer로 판정했다.
- H3: {h['H3']}. Adam 0.001과 0.01을 모두 평가했을 때만 판정했다.

## 해석 주의점

`stab`은 분류 입력에 쓰면 정답과 사실상 같은 정보를 주는 누출이므로 제외했다. `p1`은 `-(p2+p3+p4)`로 복원되는 중복 정보라 제외했지만, 제거가 반드시 성능 향상을 뜻하지는 않는다. 목표 2의 MSE와 BCE 절댓값은 척도가 달라 직접 비교하지 않았다. 2-fold 표준편차는 정밀한 신뢰도 추정이 아니며, 후보 선택에도 개발 데이터를 사용했으므로 최종 일반화 평가는 테스트 결과가 담당한다. 최종 모델은 8,000개, 목표 2 모델은 6,400개로 학습했으므로 향상을 optimizer 효과만으로 돌릴 수 없다. bootstrap은 고정된 모델과 테스트 표본의 불확실성만 반영하며 학습 seed 변동은 포함하지 않는다.

## 탐색 비용과 trade-off

적응형 4+3+4+1은 optimizer 폭, 우승 optimizer의 학습률 정밀화, 상위 2개에 대한 제한적 fold 신뢰도 확인을 배분했다. 첫 단계에서 optimizer별 학습률이 하나뿐이라 다른 학습률에서 강한 optimizer를 놓칠 수 있다. 따라서 결과는 전역 최적이 아니라 **주어진 예산과 후보에서 선택한 조합**이다. 목표 3 측정 학습 시간 합은 {comp['goal3_measured_seconds']:.1f}초다. 고정 예시 학습률 5개 `[0.0001,0.001,0.003,0.01,0.1]`의 전수 3-fold는 60회, 최종 재학습 포함 61회다. 호출 수 단순 비례 추정은 약 {comp['rough_exhaustive_seconds_scaled_by_call_count']:.1f}초지만, 조기 종료 epoch·표본 수·CPU/GPU가 달라 실제 시간은 크게 달라질 수 있다.
"""
    (OUT/"korean_results.md").write_text(report,encoding="utf-8")
    slides=f"""# 클로드 전달용 10페이지 슬라이드 구성안

1. **제목/목적** — 전력망 안정성: 회귀와 분류 비교, 12회 안의 optimizer 선택. 핵심 질문 두 개를 제시.
2. **문제 정의와 사전 가설** — 회귀 `stab>0`, 분류 `p>=0.5`; H1={h['H1']}, H2={h['H2']}, H3={h['H3']}. `pre_registered_plan.md` 문구 유지.
3. **데이터/전처리** — {a['rows']:,}행, {a['columns']}열, stable {a['class_counts'].get('stable'):,}, unstable {a['class_counts'].get('unstable'):,}; 누출 `stab`과 중복 `p1` 제외, 11개 입력, 8:2 층화 분할. 그림: `figures/data_distribution.png`.
4. **공통 모델과 조건** — 11→64 ReLU→32 ReLU→1, Adam 0.001, batch 64, 50 epoch, 동일 초기 은닉 가중치/배치 순서, scaler는 학습에만 fit.
5. **회귀 vs 분류** — 테스트 정확도 회귀 {tm['goal2_regression']['accuracy']:.4f}, 분류 {tm['goal2_classification']['accuracy']:.4f}. 그림: `figures/goal2_learning_curves.png`, 두 confusion PNG. 표: `test_metrics.csv`.
6. **오판정/불확실성** — 경계 거리별 오류, 회귀 산점도. 분류-회귀 정확도 차이 {s['bootstrap']['observed']:.4f}, 95% CI [{s['bootstrap']['ci95_percentile'][0]:.4f},{s['bootstrap']['ci95_percentile'][1]:.4f}]. 그림: `figures/misclassification_analysis.png`; H1 회고.
7. **4+3+4+1 계획** — 4 optimizer 대표점→우승 optimizer 3 LR→상위2×2-fold→최종. 폭/신뢰도/비용 trade-off와 전역 최적 아님을 명시.
8. **12회 로그와 fold** — 실제 {s['goal3_fit_calls']}/12회. 표: `goal3_run_log.csv`, `candidate_ranking.csv`, `cross_validation_results.csv`, `cross_validation_summary.csv`.
9. **최종 결과/비용/한계** — {s['final_optimizer']} lr={s['final_learning_rate']:g}, {s['final_epochs']} epoch; 최종 정확도 {tm['final']['accuracy']:.4f}. 61회 전수 탐색과 비교, 2-fold와 학습량 차이 한계.
10. **H2/H3·AI·참고문헌** — H2={h['H2']}, H3={h['H3']}. AI 활용: 계획을 실행 코드로 변환; 직접 검증/수정 사례: 과제의 12회 제한에 최종 재학습 포함, scaler fold별 재fit, 회귀 accuracy 직접 계산, 목표 3 카운터 강제. 과제 PDF와 전달된 4개 연구 링크 기재.
"""
    (OUT/"slide_outline_for_claude.md").write_text(slides,encoding="utf-8")

if __name__=="__main__": main()
