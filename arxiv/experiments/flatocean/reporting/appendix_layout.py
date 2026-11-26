"""Re-render active appendix plots with square, equally sized data panels.
Existing plot functions supply unchanged measurements. Their manuscript writes
are suppressed; only active appendix figure assets are replaced.
"""
from pathlib import Path
import re,json,math,runpy,traceback,shutil,os,textwrap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from . import style as S
seen=set();wanted={}

original_save=Figure.savefig;original_write=Path.write_text
records={};errors={}

def clean(t):
 t=re.sub(r'\b[Bb]ranches\b','Invariant spaces',t);t=re.sub(r'\b[Bb]ranch\b','Invariant space',t)
 t=t.replace('Full-block','Same-layer').replace('full-block','same-layer').replace('Block','Layer').replace('block','layer')
 return t

def standard(fig,name):
 if hasattr(fig,'_appendix_layout'):return
 axes=[a for a in fig.axes if a.get_visible() and a.axison and a.get_label()!='<colorbar>' and not hasattr(a,'_colorbar')]
 # Twin axes share a data box; retain their overlay.
 unique=[];twins=[]
 for ax in axes:
  matches=[u for u in unique if tuple(ax.get_position().bounds)==tuple(u.get_position().bounds)]
  if matches:twins.append((ax,matches[0]))
  else:unique.append(ax)
 axes=unique;n=len(axes)
 if not n:return
 cols=min(n,4)
 if n in [5,6,9]:cols=3
 try:
  oldcols=max(a.get_subplotspec().get_gridspec().ncols for a in axes)
  if oldcols<=3:cols=min(cols,oldcols)
 except Exception:pass
 rows=math.ceil(n/cols)
 side=1.12;gx=.62;gy=.88;left=.57;right=.20;top=.48
 cbaxes=[a for a in fig.axes if hasattr(a,'_colorbar')]
 horizontal=[a for a in cbaxes if a._colorbar.orientation=='horizontal']
 # Collect legends into a shared key, outside all data areas.
 handles=[];labels=[]
 for leg in list(fig.legends)+[a.get_legend() for a in axes if a.get_legend()]:
  for h,t in zip(leg.legend_handles,leg.get_texts()):
   lab=clean(t.get_text())
   if lab not in labels:handles.append(h);labels.append(lab)
  if leg in fig.legends:fig.legends.remove(leg)
  for a in axes:
   if a.get_legend() is leg:a.legend_=None
 nc=min(4,len(labels)) or 1;lr=math.ceil(len(labels)/nc);bottom=.66+.19*lr+(.43 if horizontal else 0)+(.28 if fig._supxlabel else 0)+(.58 if name=='nc_pairings' else 0)
 width=left+cols*side+(cols-1)*gx+right;height=top+rows*side+(rows-1)*gy+bottom
 fig.set_size_inches(width,height)
 for i,ax in enumerate(axes):
  row,col=divmod(i,cols);x=(left+col*(side+gx))/width;y=(bottom+(rows-1-row)*(side+gy))/height
  ax.set_position([x,y,side/width,side/height]);ax.set_box_aspect(1);ax.set_axes_locator(None)
  ax.tick_params(labelsize=6.4,pad=2);ax.tick_params(axis="x",labelsize=7.5);ax.xaxis.label.set_fontsize(7);ax.yaxis.label.set_fontsize(7)
  for title in [ax.title,ax._left_title,ax._right_title]:
   t=clean(title.get_text())
   if len(t)>26 and '\n' not in t and '$' not in t:t='\n'.join(textwrap.wrap(t,25))
   title.set_text(t);title.set_fontsize(7.3)
  for t in ax.texts:t.set_text(clean(t.get_text()));t.set_fontsize(min(t.get_fontsize(),6.5))
 for twin,owner in twins:twin.set_position(owner.get_position());twin.set_box_aspect(1)
 for cbax in cbaxes:
  cb=cbax._colorbar;cbax.set_axes_locator(None)
  if cb.orientation=='horizontal':
   j=horizontal.index(cbax);span=(width-left-right)/len(horizontal);cbax.set_position([(left+j*span+.10)/width,(.46+.19*lr)/height,(span-.25)/width,.065/height]);cbax.tick_params(labelsize=7.5);cbax.xaxis.label.set_fontsize(7)
  elif cb.mappable.axes in axes:
   a=cb.mappable.axes;p=a.get_position();cbax.set_position([p.x1+.025/width,p.y0,.05/width,p.height]);cbax.tick_params(labelsize=5.8)
 if name=='width_channels':
  axes[0].set_xlabel('Active sample--position pairs');axes[1].set_xlabel('Channels included (fraction)')
  axes[0].set_title('Channels at seed 0',loc='left',fontsize=7.3);axes[1].set_title('Least-active channels first',loc='left',fontsize=7.3)
 if name=='background_trajectory':
  axes[0].set_xticks([0,1e3,1e6]);axes[1].set_xticks([0,1e2,1e4]);axes[2].xaxis.labelpad=12
 if name=='paper_sweep_split':
  for a in [axes[i] for i in [0,1,3,4]]:
   a.tick_params(axis='x',pad=18)
   for t in a.texts:
    if re.fullmatch(r'L[0-9]+',t.get_text()):t.set_rotation(90);t.set_ha('center');t.set_va('top');t.set_fontsize(7)
 if name=='margin_dists':
  axes[1].set_xticklabels(['Control',r'$\lambda_0$',r'$\lambda_1$'])
 if name=='p1_exposure_consistent':
  for a,t in zip(axes,['Invariant-space rays','Excluded-input response','Atomic kernel dimension']):a.set_title(t,loc='left',fontsize=7.3)
 if name=='nc_pairings':
  for a in axes:
   for i,t in enumerate(a.texts):t.set_position((0,-.46-i*.15))
 # Resolve actual tick-label collisions at the final physical panel size.
 from matplotlib.ticker import FixedLocator, NullFormatter
 fig.canvas.draw()
 for ax in axes:
  ax.xaxis.set_minor_formatter(NullFormatter())
  def visible_ticks():
   lo,hi=sorted(ax.get_xlim())
   return [(loc,t) for loc,t in zip(ax.get_xticks(),ax.get_xticklabels()) if lo<=loc<=hi and t.get_visible() and t.get_text()]
  def crowded():
   bb=sorted([t.get_window_extent(fig.canvas.get_renderer()) for _,t in visible_ticks()],key=lambda b:b.x0)
   return any(b.x1+3>c.x0 for b,c in zip(bb,bb[1:]))
  if crowded():
   ticks=visible_ticks()
   numeric=all(not re.search(r'[A-Za-z]',re.sub(r'\\[A-Za-z]+','',t.get_text()).replace('mathdefault','')) for _,t in ticks)
   if numeric and len(ticks)>3:
    inds=sorted(set([0,(len(ticks)-1)//2,len(ticks)-1]))
    locs=[ticks[j][0] for j in inds];labs=[ticks[j][1].get_text() for j in inds]
    ax.xaxis.set_major_locator(FixedLocator(locs));ax.set_xticklabels(labs,fontsize=7.5)
   else:
    if len(ticks)>8:
     ticks=ticks[::2]+([ticks[-1]] if (len(ticks)-1)%2 else [])
     ax.xaxis.set_major_locator(FixedLocator([loc for loc,_ in ticks]))
     ax.set_xticklabels([t.get_text() for _,t in ticks],fontsize=7.5)
    if any('\n' in t.get_text() for _,t in ticks):
     ax.xaxis.set_major_locator(FixedLocator([loc for loc,_ in ticks]))
     ax.set_xticklabels([' · '.join(t.get_text().splitlines()) for _,t in ticks],fontsize=7.5)
    for angle in [45,65,90]:
     for _,t in visible_ticks():t.set_rotation(angle);t.set_ha('right' if angle<90 else 'center');t.set_rotation_mode('default');t.set_va('top')
     if not crowded():break
  ax.tick_params(axis='x',labelsize=7.5)
 fig.canvas.draw()
 for t in fig.findobj(Text):t.set_text(clean(t.get_text()))
 if fig._supylabel:fig._supylabel.set_position((.003,.53));fig._supylabel.set_fontsize(7)
 if fig._supxlabel:fig._supxlabel.set_position((.5,(bottom-.37)/height));fig._supxlabel.set_fontsize(7)
 if handles:fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.015),ncol=nc,fontsize=6.4,handlelength=1.7,columnspacing=.8,labelspacing=.35)
 fig.canvas.draw();dims=[(a.get_window_extent().width/fig.dpi,a.get_window_extent().height/fig.dpi) for a in axes]
 assert all(abs(x-y)<1e-6 for x,y in dims),(name,dims)
 fig._appendix_layout=True;records[name]=dict(panels=n,columns=cols,rows=rows,axis_inches=dims,width=width,height=height,x_ticks=[[t.get_text() for t in a.get_xticklabels() if t.get_visible()] for a in axes],x_tick_rotations=[[t.get_rotation() for t in a.get_xticklabels() if t.get_visible()] for a in axes])




