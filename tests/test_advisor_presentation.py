"""Presentation-only localization; shared semantic data, no Advisor execution."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_advisor_planning as planner_fixture
import test_advisor_goals as goal_fixture
from hyso.advisor import (AdvisorError, AdvisorReport, AdvisorRequest, EvidenceRef, Finding,
                          NativeObservation, RuleVersion)
from hyso.advisor.engine import _advise_validated
from hyso.advisor.presentation import build_display_model
from hyso.advisor.render import render_markdown


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=planner_fixture.PlanningTests
        fixture.setUpClass(); cls.addClassCleanup(fixture.doClassCleanups)
        cls.report=fixture.result
        from hyso.advisor import advise
        candidate_source,_=fixture().throughput_source()
        cls.candidate=advise(candidate_source)
        goal=goal_fixture.GoalEngineTests().goal(goal_id='doğruluk-İnceleme',target=.001)
        cls.goals=_advise_validated(fixture.resolver,AdvisorRequest([goal]),plan_experiments=True)
        cls.before=cls.report.to_json(); cls.before_id=cls.report.report_id
        def forbidden(*a,**k):raise AssertionError('Presentation must not run any engine or native resolver')
        cls.guards=ExitStack(); cls.addClassCleanup(cls.guards.close)
        for name in ('advise','_advise_validated'):
            cls.guards.enter_context(patch('hyso.advisor.engine.'+name,forbidden))
        for name in ('evaluate_goal','evaluate_goals'):
            cls.guards.enter_context(patch('hyso.advisor.goals.'+name,forbidden))
        cls.guards.enter_context(patch('hyso.advisor.resolver.NativeMetricResolver.__init__',forbidden))

    def test_all_four_markdown_views(self):
        for language in ('tr','en'):
            for audience in ('summary','technical'):
                with self.subTest(language=language,audience=audience):
                    text=self.report.to_markdown(language=language,audience=audience)
                    self.assertTrue(text.startswith('# HYSo'))
                    self.assertIn('0,86681' if language=='tr' else '0.86681',text)
                    if audience=='technical':
                        for field in ('source_report_id','metric_path','bindings','rule_versions','protocol_json'):
                            self.assertIn(field,text)

    def test_tr_summary_explanations_are_turkish_not_heading_only(self):
        text=self.report.to_markdown(language='tr',audience='summary')
        for phrase in ('Mevcut kanıtlar henüz bir optimizasyon deneyi önermek için yeterli değil.',
                       'mevcut P03/P04 işlem hattı ölçümlerini inceleyin',
                       'Modelin kendi gecikmesi ile uçtan uca süre',
                       'Bu raporlama sırasında model üzerinde herhangi bir değişiklik'):
            self.assertIn(phrase,text)
        self.assertNotIn('Inspect retained pipeline',text)
        self.assertNotIn('No optimization experiment',text)

    def test_en_summary_is_english(self):
        text=self.report.to_markdown(language='en',audience='summary')
        self.assertIn('Current model performance',text)
        self.assertIn('No optimization experiment is sufficiently supported yet.',text)
        self.assertNotIn('Mevcut kanıtlar',text)

    def test_default_markdown_exact_legacy_path(self):
        self.assertEqual(self.report.to_markdown(),render_markdown(self.report))
        self.assertEqual(self.report.to_markdown(language=None,audience=None),render_markdown(self.report))

    def test_display_data_same_across_languages_and_audiences(self):
        data=build_display_model(self.report)
        for language in ('tr','en'):
            for audience in ('summary','technical'):
                view=self.report.to_display_dict(language=language,audience=audience)
                self.assertEqual(view['data'],data)
                self.assertEqual(view['source_advisor_report_id'],self.before_id)
                self.assertTrue(view['presentation_only'])
                self.assertFalse(view['authoritative_evidence'])
        self.assertNotEqual(self.report.to_display_dict(language='tr')['text'],self.report.to_display_dict(language='en')['text'])

    def test_canonical_json_ids_values_statuses_unchanged(self):
        for language in ('tr','en'):
            self.report.to_markdown(language=language)
            self.report.to_display_json(language=language)
        self.assertEqual(self.report.to_json(),self.before)
        self.assertEqual(self.report.report_id,self.before_id)
        self.assertEqual(self.report.to_display_dict(language='tr')['data']['original_technical_record'],json.loads(self.before))

    def test_display_not_accepted_as_canonical_evidence(self):
        view=self.report.to_display_dict(language='tr')
        self.assertEqual(view['format'],'hyso.advisor.display')
        with self.assertRaises(AdvisorError):AdvisorReport.from_dict(view)

    def test_explicit_utf8_exports_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            for language in ('tr','en'):
                md=Path(directory)/(language+'.md'); js=Path(directory)/(language+'.json')
                self.report.to_markdown(md,language=language,audience='summary')
                self.report.to_display_json(js,language=language,audience='summary')
                self.assertEqual(json.loads(js.read_bytes().decode('utf-8')),self.report.to_display_dict(language=language))
                self.assertEqual(md.read_text(encoding='utf-8'),self.report.to_markdown(language=language))
                with self.assertRaises(FileExistsError):self.report.to_display_json(js,language=language)
                with self.assertRaises(FileExistsError):self.report.to_markdown(md,language=language)

    def test_turkish_characters_retained_in_original_and_localized_text(self):
        text=self.goals.to_markdown(language='tr',audience='technical')
        for char in 'ğüşıöçİ':self.assertIn(char,text)
        encoded=self.goals.to_display_json(language='tr').encode('utf-8')
        self.assertIn('doğruluk-İnceleme',encoded.decode('utf-8'))

    def test_unsupported_language_and_audience_no_file_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.md'
            for options in (dict(language='fr'),dict(language='TR'),dict(language=''),dict(language=7),
                            dict(language='tr',audience='marketing'),dict(audience='')):
                with self.subTest(options=options),self.assertRaises(AdvisorError):
                    self.report.to_markdown(path,**options)
                self.assertFalse(path.exists())
            with self.assertRaisesRegex(AdvisorError,'unsupported_language'):
                self.report.to_display_json(path,language='de')

    def test_unknown_rule_or_version_is_explicit_original(self):
        original=next(f for f in self.report.findings if f.rule_id=='plan.latency-stage')
        for rule,version in (('future.rule','1'),('plan.latency-stage','99')):
            f=replace(original,rule_id=rule,rule_version=version,statement='Original English free text; do not translate blindly.')
            report=replace(self.report,findings=(f,),rule_versions=(RuleVersion(rule,version),))
            view=report.to_display_dict(language='tr')
            self.assertTrue(view['text']['unknown_records'])
            self.assertEqual(view['text']['actions'],[])
            technical=report.to_markdown(language='tr',audience='technical')
            self.assertIn('Özgün teknik kayıt',technical)
            self.assertIn('Original English free text',technical)
            self.assertNotIn('mevcut P03/P04 işlem hattı',report.to_markdown(language='tr'))

    def test_unknown_untyped_metric_is_not_guessed_or_zero(self):
        ref=EvidenceRef('a'*64,'b'*64,'measurements.value','c'*64,'d'*64,'unknown_domain','mystery_unit','unknown',())
        finding=Finding('untyped','OBSERVATION','Unknown latency or FPS text',evidence=(ref,))
        report=AdvisorReport('a'*64,'e'*64,AdvisorRequest(),findings=(finding,))
        view=report.to_display_dict(language='tr')
        self.assertEqual(view['data']['metrics'],[])
        self.assertIn('sayısal anlam tahmin edilmedi',view['text']['unknown_records'][0]['message'])
        self.assertIn('Eksik değerler sıfır kabul edilmedi',report.to_markdown(language='tr'))

    def test_unknown_typed_metric_rejected_not_localized(self):
        source=AdvisorReport.from_dict(self.report.to_dict())
        f=next(f for f in source.findings if type(f) is NativeObservation)
        object.__setattr__(f.resolved_metric,'metric','future_metric')
        with self.assertRaises(AdvisorError):source.to_display_dict(language='tr')

    def test_fraction_units_and_measurement_domains_same(self):
        tr=self.report.to_display_dict(language='tr'); en=self.report.to_display_dict(language='en')
        self.assertEqual(tr['data']['metrics'],en['data']['metrics'])
        by={m['metric']:m for m in tr['text']['metrics']}
        self.assertEqual(by['test_accuracy']['value'],'%75')
        self.assertEqual(by['completed_images_per_second']['value'],'12.442,17 görüntü/sn')
        self.assertIn('Model P95',by['resident_forward_p95']['label'])
        self.assertIn('Uçtan uca',by['e2e_p95']['label'])
        self.assertIn('süreç RSS',by['process_rss_sampled_peak']['label'])
        self.assertIn('süreç RAM kullanımı değildir',by['model_parameter_buffer_logical_bytes']['label'])

    def test_goal_hard_statuses_and_plan_authorization_same(self):
        tr=self.goals.to_display_dict(language='tr'); en=self.goals.to_display_dict(language='en')
        self.assertEqual(tr['data']['goals'],en['data']['goals'])
        self.assertEqual(tr['data']['findings'],en['data']['findings'])
        self.assertEqual(tr['data']['goals'][0]['evaluation']['observed'],'observed_missed')
        self.assertEqual(tr['data']['goals'][0]['goal']['constraint'],'hard')
        self.assertEqual(tr['data']['counts']['VERIFIED_IMPROVEMENT'],0)
        for f in tr['data']['findings']:
            if f['plan']:self.assertEqual(f['plan']['authorization'],'not_authorized')

    def test_acceptance_boundaries_localized_and_originals_preserved(self):
        view=self.report.to_display_dict(language='tr')
        risks=' '.join(view['text']['risks'])
        for text in ('B4 native çökme riski açık','P18','P29','P31 7200'):
            self.assertIn(text,risks)
        self.assertEqual(view['data']['original_risks'],list(self.report.risks))

    def test_display_is_deterministic_and_detached(self):
        first=self.report.to_display_json(language='tr')
        self.assertEqual(first,self.report.to_display_json(language='tr'))
        detached=self.report.to_display_dict(language='tr'); detached['data']['metrics'].clear()
        self.assertTrue(self.report.to_display_dict(language='tr')['data']['metrics'])

    def test_conditional_candidate_is_localized_without_authorization_or_gain(self):
        tr=self.candidate.to_display_dict(language='tr')
        en=self.candidate.to_display_dict(language='en')
        self.assertEqual(tr['data'],en['data'])
        candidate=next(a for a in tr['text']['actions'] if a['candidate'])
        self.assertIn('yalnız batch boyutunu',candidate['action'])
        self.assertIn('Yetkilendirilmedi',candidate['authorization'])
        self.assertIn('Beklenen kazanç ve maliyet bilinmiyor',self.candidate.to_markdown(language='tr'))


if __name__=='__main__':unittest.main(verbosity=2)
