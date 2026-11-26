"""Regenerate exactly the published figure assets from saved measurements.

Output goes into build/figures. Paper sources and result records are read-only.
"""
from pathlib import Path
import argparse, os, sys, json, runpy, copy, shutil
from contextlib import contextmanager


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/figures'))
    parser.add_argument('--only', nargs='+', help='recipe names from --list')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    catalog = json.loads((root/'configs/figures.json').read_text())
    if args.list:
        for name in sorted({v['recipe'] for v in catalog.values()}): print(name)
        return 0
    recipes={v['recipe'] for v in catalog.values()}
    if args.only and set(args.only)-recipes:
        parser.error('unknown recipe: '+', '.join(sorted(set(args.only)-recipes)))
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    work = output/'intermediates'; work.mkdir(exist_ok=True)
    os.environ['FLATOCEAN_FIGURES_DIR'] = str(work)
    os.environ['FLATOCEAN_TABLES_DIR'] = str(work/'tables')
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0,str(root/'scripts/analysis'))
    EXP = root
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from flatocean.reporting import appendix_layout as layout
    from flatocean.reporting import section_c as C, appendix_polish as P
    from flatocean.reporting import appendix_review as A, appendix_comments as AC
    from flatocean.reporting import comment_figures as CF, applications_lmc as L
    from flatocean.reporting import appendix_finish as F
    from flatocean.reporting import main_polish as M, atomic_scaling, radius_samples
    from flatocean.reporting import vit_oceans, loss_geometry, newinput_interpolation, aistats_figures
    from flatocean.reporting import cooccurrence_schematic, curvature_rays
    import plot_ray_extensions
    
    
    generated={};current='summary_inputs'
    original_save = Figure.savefig
    
    def save(fig, filename, *pos, **kw):
        path=Path(filename); stem=path.stem
        if path.suffix!='.pdf':return
        if stem in {'compact_tails','compact_selective','persistence_summary'}:
            step=6 if stem=='persistence_summary' else 12
            axes=[a for a in fig.axes if a.get_visible() and a.axison and not hasattr(a,'_colorbar')]
            for start in range(0,len(axes),step):
                part=copy.deepcopy(fig)
                for i,ax in enumerate([a for a in part.axes if a.get_visible() and a.axison and not hasattr(a,'_colorbar')]):
                    ax.set_visible(start<=i<start+step)
                save(part,work/f'{stem}_part{start//step+1}.pdf',*pos,**kw)
                plt.close(part)
            return
        if stem not in catalog:return
        if catalog[stem]['appendix'] and stem not in {'cnn_units_wide','lmc_planes_mlp','lmc_planes_cnn'}:
            layout.standard(fig,stem)
        # A plotting function may write only inside this invocation's build directory.
        if not path.resolve().is_relative_to(output):
            raise RuntimeError(f'Plot tried writing outside its output directory: {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        original_save(fig,path,*pos,**kw)
        final=output/f'{stem}.pdf'
        if path.resolve()!=final:shutil.copy2(path,final)
        generated[stem]=current
    
    Figure.savefig=save
    try:
        data = C.collect()
        C.OUT.mkdir(parents=True,exist_ok=True)
        for name,rows in zip(['atomic','selective','branches','neuronwise'],data):
            C.write_csv(name,rows)
        C.branch_attempt_figures()
        C.response_figures(data[2])
        C.response_figures(data[2],margin=True)
        def margin_rays():
            data={}
            for r in P.records('margin_rays'):
                key=(r['model'],int(r['seed']),r['stage'],r['direction_set'],r['mode'],float(r['radius']))
                data[key]={k:float(r[k]) for k in ['true','frozen','ten_percent']}
            C.render_responses(data,True)
        
        tasks = [
         ('atomic',lambda:C.atomic_figures(data[0])),('margin_rays',margin_rays),
         ('selective',lambda:C.selective_figures(data[1],data[0],data[3])),
         ('construction',lambda:C.render_constructions(P.records('construction_attempts'))),
         ('transition',C.transition_figures),('transition_details',C.transition_details),
         ('background',C.background_figure),('square_sweeps',C.square_sweeps),
         ('atomic_rays',A.atomic_rays),('activation_rays',A.activation_rays),
         ('residual_rays',A.residual_rays),('cnn_units',AC.cnn_units),
         ('cluster_stages',P.cluster_stages),('clusters',AC.clustering),('nc',P.nc),
         ('data_dimensions',P.data_dimensions),('data_rays',P.ray_grid),('p1_summary',P.p1_summary),
         ('persistence',A.persistence),('persistence_tails',A.persistence_tails),
         ('persistence_diagnostics',A.persistence_diagnostics),('xray',P.xray_catalogue),
         ('clustering_seeds',CF.clustering),('nc_pairings',CF.nc_pairings),
         ('lmc_segments',lambda:L.cohorts_and_curves(L.segments())),
         ('lmc_planes',L.landscapes),('lmc_gates',lambda:L.gates(L.segments())),
         ('vit_joint',vit_oceans.main),('hessian',loss_geometry.main),
         ('newinput',newinput_interpolation.main),('margins',aistats_figures.fig_margins),
         ('new_evaluation',F.evaluation),('new_families',F.family_figures),
         ('new_certificates',F.certificate_figures),('hero',CF.hero_shared_axes),
         ('main_transition',CF.transition),('main_atomic',CF.atomic),
         ('activation_tails',M.activations),('frozen_terms',M.polynomial),
         ('atomic_scaling',atomic_scaling.main),('cooccurrence',cooccurrence_schematic.main),
         ('curvature',curvature_rays.main),('gelu',plot_ray_extensions.gelu),
        ]
        for name in ['soft_selectivity_fig','cnn_width_fig','dmap_pca','sweep_split','deficit_mechanisms_fig','margin_dists']:
            tasks.append((name,lambda name=name:runpy.run_path(str(EXP/'paper_figures'/f'{name}.py'),run_name='__main__')))
        tasks.append(('radius_probability',lambda:radius_samples.main(['--probability'])))
        for name,fn in tasks:
            if args.only and name not in args.only:continue
            current=name
            fn();plt.close('all');print('Rendered',name,flush=True)
    finally:
        Figure.savefig=original_save
    expected={name for name,info in catalog.items() if not args.only or info['recipe'] in args.only}
    missing=expected-generated.keys()
    (output/'manifest.json').write_text(json.dumps({'generated':generated,'missing':sorted(missing)},indent=2)+'\n')
    if missing:raise RuntimeError(f'Missing published figures: {sorted(missing)}')
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
