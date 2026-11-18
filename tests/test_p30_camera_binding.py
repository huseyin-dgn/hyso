"""CPU doubles only; no physical capture or COM device enumeration."""
from types import SimpleNamespace
import pytest
from hyso.performance import streaming
from hyso.performance import _windows_camera as metadata


BINDING = dict(index=0,backend='DSHOW',friendly_name='offline physical double',device_path='offline-usb-path')


@pytest.mark.parametrize('devices',[[],[dict(BINDING,device_path='other')],
    [dict(BINDING,index=1)], [dict(BINDING,friendly_name='other')]])
def test_identity_or_order_drift_fails_closed(monkeypatch,devices):
    monkeypatch.setattr(metadata,'enumerate_cameras',lambda:devices)
    with pytest.raises(ValueError,match='identity/order changed'):
        metadata.verify_binding(BINDING,0)


@pytest.mark.parametrize('binding',[None,{},dict(BINDING,backend='MSMF'),dict(BINDING,device_path=None)])
def test_no_implicit_backend_or_virtual_device(binding):
    with pytest.raises(ValueError):metadata.validate_binding(binding,0)


@pytest.mark.parametrize('failure',['before_open','after_open','backend','none'])
def test_capture_binding_brackets_open_and_preserves_cleanup(monkeypatch,tmp_path,failure):
    import cv2
    calls=[];closed=[];headers=[]
    def verify(binding,index):
        calls.append('verify')
        if failure=='before_open' or (failure=='after_open' and len(calls)==3):
            raise ValueError('identity/order changed')
        return dict(BINDING)
    class Capture:
        def isOpened(self):return True
        def get(self,key):return 30.
        def getBackendName(self):return 'MSMF' if failure=='backend' else 'DSHOW'
        def release(self):closed.append('capture')
    def create(index,backend):
        assert (index,backend)==(0,cv2.CAP_DSHOW)
        calls.append('open');return Capture()
    monkeypatch.setattr(metadata,'verify_binding',verify)
    monkeypatch.setattr(cv2,'VideoCapture',create)
    monkeypatch.setattr(streaming,'_Journal',lambda root,header,*a:headers.append(header))
    monkeypatch.setattr(streaming,'_atomic',lambda *a:None)
    processor=SimpleNamespace(identity={},close=lambda:closed.append('processor'))
    source=streaming.StreamSource('camera',0,'offline double',camera_binding=BINDING)
    if failure!='none':
        with pytest.raises(ValueError):
            streaming._prepare_capture(tmp_path,dict(token='test'),streaming.StreamProtocol(),source,processor,{})
        assert closed==(['processor'] if failure=='before_open' else ['capture','processor'])
    else:
        streaming._prepare_capture(tmp_path,dict(token='test'),streaming.StreamProtocol(),source,processor,{})
        assert calls==['verify','open','verify']
        assert headers[0]['capture']['camera_binding']==BINDING


def test_handler_rejects_unbound_camera_before_measurement(monkeypatch,tmp_path):
    from tools.full_acceptance import handlers
    import hyso.performance
    monkeypatch.setattr(hyso.performance,'measure_streaming',lambda *a,**k:pytest.fail('unbound capture attempted'))
    m=dict(optional=dict(camera_adapter=dict(configuration=dict(camera_provenance='name only'))))
    with pytest.raises(ValueError,match='binding required'):handlers.camera(m,tmp_path,live=True)
