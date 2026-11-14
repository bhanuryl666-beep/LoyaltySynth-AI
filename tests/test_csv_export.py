import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app as web_app


class CsvExportTests(unittest.TestCase):
    def test_generate_route_and_csv_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dataset_path = directory / "transactions.csv"
            with patch.object(web_app, "DATA_DIR", directory), patch.object(web_app, "DATA_FILE", dataset_path):
                client = web_app.app.test_client()
                response = client.post("/generator", data={
                    "count": "100",
                    "customers": "20",
                    "start_date": "2025-01-01",
                    "end_date": "2025-02-01",
                    "earning_ratio": "1.25",
                    "fraud_percent": "10",
                    "segments": ["Standard", "Premium", "VIP"],
                    "mode": "rules",
                })
                self.assertEqual(response.status_code, 302)
                download = client.get("/download")
                self.assertEqual(download.status_code, 200)
                self.assertIn("text/csv", download.content_type)
                self.assertIn(b"transaction_id", download.data)
                download.close()
                exported = pd.read_csv(dataset_path)
                self.assertEqual(len(exported), 100)
                self.assertIn("earning_ratio", exported.columns)


if __name__ == "__main__":
    unittest.main()
