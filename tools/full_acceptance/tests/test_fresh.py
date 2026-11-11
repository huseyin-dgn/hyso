"""Fresh verdicts, selected routing and scoped auxiliary admission; no native work."""
from test_infrastructure import Fixture
from full_acceptance import fresh, reporting, core, cli


class FreshAcceptance(Fixture):
    def select(self):
        self.m['model_run']=dict(arm='A')
        self.m['assets']=dict(models=dict(A=dict(sha256='a'*64)))
        self.m['fresh_acceptance']=dict(feature_scopes={})
        self.m['protocols']=dict(inference=dict(precisions_cpu=['fp32','fp64'],precisions_cuda=['fp32','amp_fp16']))

    def test_old_passes_never_promote_unexecuted_features(self):
        rows=self.old_matrix();self.select()
        values=reporting.matrix(self.m,rows)
        self.assertEqual(len(values),35)
        self.assertTrue(all(f['decision']=='NOT_RUN' for f in values))
        self.assertEqual(fresh.arms(self.m),['A'])

    def test_fresh_mapping_requires_new_capacity_and_real_capture(self):
        self.assertEqual(fresh.MAPPING['P17'],['03.cpu','04.cuda'])
        self.assertEqual(fresh.MAPPING['P14'],['06.p13_capture','06.layer'])
        self.assertFalse(any('12.replay' in x for x in fresh.MAPPING.values()))

    def test_auxiliary_is_bound_to_selected_arm_parent_and_output(self):
        self.select();root=self.out/'stages/03.cpu/A/tradeoff-trial-1'
        fresh.authorize_auxiliary(self.m,'__tradeoff_1','tradeoff_trial',root)
        with self.assertRaises(PermissionError):
            fresh.authorize_auxiliary(self.m,'__tradeoff_1','tradeoff_trial',self.out/'stages/03.cpu/B/tradeoff-trial-1')
        next(e for e in self.m['experiments'] if e['id']=='03.cpu')['authorization']['approved']=False
        with self.assertRaises(PermissionError):
            fresh.authorize_auxiliary(self.m,'__tradeoff_1','tradeoff_trial',root)

    def test_capture_finalizer_cannot_acquire_native_approval(self):
        self.select()
        with self.assertRaises(PermissionError):
            fresh.authorize_auxiliary(self.m,'__p13_finalize_stage','p13_finalize_stage',self.out/'preparation/p13-stage-finalizer')
        self.assertEqual(cli._fatal_experiment_boundary(dict(status='FAIL',supervisor=dict(
            reason='resource_headroom_insufficient',native_crash=False,cleanup=dict(active_processes=0)))),
            'resource_headroom_insufficient')

    def test_successful_stage_without_measurement_is_partial(self):
        rows=self.old_matrix();self.select()
        for ident in ('03.cpu','04.cuda'):
            folder=self.out/ident;folder.mkdir()
            rows[ident].update(status='PASS',attempted=True,evidence_directory=str(folder),
                supervisor=dict(exit_code=0,cleanup=dict(active_processes=0)))
            next(e for e in self.m['experiments'] if e['id']==ident)['authorization']['approved']=True
        values={r['id']:r for r in reporting.matrix(self.m,rows)}
        self.assertEqual(values['P01']['decision'],'PARTIAL')
        self.assertTrue(values['P01']['decision_basis'])
