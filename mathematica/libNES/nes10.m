(* ::Package:: *)

(************************************************************************)
(* This file was generated automatically by the Mathematica front end.  *)
(* It contains Initialization cells from a Notebook file, which         *)
(* typically will have the same name as this file except ending in      *)
(* ".nb" instead of ".m".                                               *)
(*                                                                      *)
(* This file is intended to be loaded into the Mathematica kernel using *)
(* the package loading commands Get or Needs.  Doing so is equivalent   *)
(* to using the Evaluate Initialization Cells menu command in the front *)
(* end.                                                                 *)
(*                                                                      *)
(* DO NOT EDIT THIS FILE.  This entire file is regenerated              *)
(* automatically each time the parent Notebook file is saved in the     *)
(* Mathematica front end.  Any changes you make to this file will be    *)
(* overwritten.                                                         *)
(************************************************************************)



sNESstep[f_,dim_,\[Mu]\[Sigma]_,\[Lambda]_,\[Eta]\[Delta]_,\[Eta]\[Sigma]_]:=Module[{\[Mu],\[Sigma],z,s,u,g\[Delta],g\[Sigma]},
{\[Mu],\[Sigma]}=\[Mu]\[Sigma];
s=RandomReal[NormalDistribution[0,1],{\[Lambda],dim}];
s=s[[Ordering[f/@((\[Mu]+\[Sigma] s\[Transpose])\[Transpose])]]];
u=utilityFunction[\[Lambda]];
{\[Mu]+\[Eta]\[Delta] \[Sigma] (u.s),\[Sigma] Exp[\[Eta]\[Sigma]/2 (u.(s^2-1))]}
]


sNES[f_,dim_,\[Mu]\[Sigma]_,\[Lambda]_,\[Eta]\[Delta]_,\[Eta]\[Sigma]_,nIter_]:=Nest[sNESstep[f,dim,#,\[Lambda],\[Eta]\[Delta],\[Eta]\[Sigma]]&,\[Mu]\[Sigma],nIter]


populationSize[d_]:=4+Floor[3 Log[d]]


utilityFunction[n_]:=utilityFunction[n]=Reverse[N[Max[0,#]&/@(Log[n/2-1]-Log[Range[n]])/Total[Max[0,#]&/@(Log[n/2-1]-Log[Range[n]])]-1/n]]


learningRateSNES[d_]:=N[(3+Log[d])/(5Sqrt[d])]


eye[n_]:=eye[n]=IdentityMatrix[n]


xNESstep[f_,dim_,\[Mu]A_,\[Lambda]_,\[Eta]\[Mu]_,\[Eta]A_]:=Module[{\[Mu],\[Sigma],z,x,u,g\[Mu],g\[Sigma],gA,A,expA},
{\[Mu],A}=\[Mu]A;
z=RandomReal[NormalDistribution[0,1],{\[Lambda],dim}];
expA=N@MatrixExp[A];
x=(\[Mu]+expA.z\[Transpose])\[Transpose];
z=z[[Ordering[f/@x]]];
u=utilityFunction[\[Lambda]];
g\[Mu]=z\[Transpose].u;
gA=Plus@@MapThread[#1 (Outer[Times,#2,#2]-eye[dim])&,{u,z}];
{\[Mu]+\[Eta]\[Mu] expA.g\[Mu],A+(\[Eta]A/2)gA}
]


learningRateXNES[d_]:=N[3(3+Log[d])/(5d Sqrt[d])]


xNES[f_,dim_,\[Mu]A_,\[Lambda]_,\[Eta]\[Mu]_,\[Eta]A_,nIter_]:=Nest[xNESstep[f,dim,#,\[Lambda],\[Eta]\[Mu],\[Eta]A]&,\[Mu]A,nIter]



