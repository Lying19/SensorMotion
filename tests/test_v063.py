import io, sys, unittest, zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from app import app, UPLOAD_LOCK
from data_loader import parse_upload
from inference import infer
CSV=b'seconds_elapsed,acc_source,acc_x_m_s2,acc_y_m_s2,acc_z_m_s2,gyro_x_rad_s,gyro_y_rad_s,gyro_z_rad_s\n0,linear,1,0,0,0,1,0\n0,linear,3,0,0,0,3,0\n1,with_gravity,9,0,0,0,2,0\n'
class Tests(unittest.TestCase):
    def test_browser(self):
        r=parse_upload('phone.csv',CSV)
        self.assertEqual(r['quality']['duplicate_timestamps'],1)
        self.assertEqual(r['datasets']['Accelerometer']['series'][0]['values'],[2.0])
        self.assertIn('TotalAcceleration',r['datasets'])
        self.assertEqual(infer('phone.csv',CSV,None)['status'],'unavailable')
    def test_optional_label(self):
        with app.test_client() as c:
            r=c.post('/api/upload',data={'file':(io.BytesIO(CSV),'phone.csv')})
            self.assertEqual(r.status_code,200); self.assertIsNone(r.json['ground_truth'])
    def test_busy(self):
        with UPLOAD_LOCK, app.test_client() as c:
            self.assertEqual(c.post('/api/upload').status_code,503)
    def test_zip(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            for name in ['Accelerometer','Gyroscope']:
                z.writestr(name+'.csv','seconds_elapsed,x,y,z\n0,1,2,3\n1,2,3,4\n')
        self.assertEqual(parse_upload('a.zip',b.getvalue())['sensors'],['Accelerometer','Gyroscope'])
