"""Run only acceptance-tool unit/fault tests and export actual JUnit records."""
from pathlib import Path
import argparse,contextlib,json,sys,time,unittest,xml.etree.ElementTree as ET

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--junit',type=Path,required=True);parser.add_argument('--log',type=Path,required=True)
    args=parser.parse_args();args.junit.parent.mkdir(parents=True,exist_ok=True)
    if args.junit.exists() or args.log.exists():raise FileExistsError('new test artifacts required')
    rows=[]
    class Result(unittest.TextTestResult):
        def startTest(self,test):self.started=time.perf_counter();self.outcome='PASS';self.detail='';super().startTest(test)
        def addFailure(self,test,error):self.outcome='FAIL';self.detail=self._exc_info_to_string(error,test);super().addFailure(test,error)
        def addError(self,test,error):self.outcome='ERROR';self.detail=self._exc_info_to_string(error,test);super().addError(test,error)
        def addSkip(self,test,reason):self.outcome='SKIP';self.detail=reason;super().addSkip(test,reason)
        def stopTest(self,test):rows.append(dict(id=test.id(),status=self.outcome,detail=self.detail,seconds=time.perf_counter()-self.started));super().stopTest(test)
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py')
    with args.log.open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        answer=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result).run(suite)
    xml=ET.Element('testsuite',name='HYSo-acceptance-infrastructure',tests=str(len(rows)),failures=str(len(answer.failures)),errors=str(len(answer.errors)),skipped=str(len(answer.skipped)))
    for r in rows:
        case=ET.SubElement(xml,'testcase',classname='acceptance.infrastructure',name=r['id'],time=str(r['seconds']))
        if r['status']!='PASS':ET.SubElement(case,{'FAIL':'failure','ERROR':'error','SKIP':'skipped'}[r['status']],message=r['detail'])
    ET.ElementTree(xml).write(args.junit,encoding='utf-8',xml_declaration=True)
    print(json.dumps(dict(tests=len(rows),passed=len(rows)-len(answer.failures)-len(answer.errors)-len(answer.skipped),failures=len(answer.failures),errors=len(answer.errors),skips=len(answer.skipped),
        native_GPU_experiments=0,training_runs=0,real_cameras=0,long_runs=0)))
    return 0 if answer.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
