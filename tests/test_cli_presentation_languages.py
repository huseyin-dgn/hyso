"""CLI presentation uses the four languages already supported by its readers."""
import json
from pathlib import Path

import pytest

from hyso.cli import main
from hyso.cli.main import build_parser
from hyso.cli.common import CLIError

COMMANDS=(['reports','render','retained.json'],['advisor','recommend','retained.json'],
          ['compare','a.json','b.json','--baseline-model','A','--candidate-model','B','--features','P26'])


@pytest.mark.parametrize('command',COMMANDS,ids=['reports','advisor','compare'])
@pytest.mark.parametrize('language',['tr','en','de','es'])
def test_presentation_parser_accepts_supported_languages(command,language):
    args=build_parser().parse_args([*command,'--language',language])
    assert args.language==language


@pytest.mark.parametrize('command',COMMANDS,ids=['reports','advisor','compare'])
@pytest.mark.parametrize('language',['fr','TR'])
def test_presentation_parser_keeps_unknown_languages_rejected(command,language):
    with pytest.raises(CLIError) as exc:build_parser().parse_args([*command,'--language',language])
    assert exc.value.code==2


@pytest.mark.parametrize('language',['tr','en','de','es'])
@pytest.mark.parametrize('fmt,suffix',[('json','json'),('markdown','md'),('html','html')])
def test_CLI_renders_retained_training_in_each_supported_language(language,fmt,suffix,tmp_path,capsys):
    source=Path(__file__).resolve().parents[1]/'examples/reporting/fixtures/training-result-v1.json'
    output=tmp_path/('report.'+suffix)
    assert main(['reports','render',str(source),'--language',language,'--format',fmt,'--output',str(output),'--json'])==0
    envelope=json.loads(capsys.readouterr().out)
    assert envelope['result']['language']==language
    assert output.stat().st_size>100
    if fmt=='json':assert json.loads(output.read_text(encoding='utf-8'))['language']==language
