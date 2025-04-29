# Robust DeID – De-Identification of Medical Notes with Transformer Architectures

## Files

* **Forward_Pass.ipynb** – main de-identification pipeline  
  * All edited cells are marked with `# 🔄 MODIFIED`.

* **formatting.ipynb** – converts a CSV of mental-imagery notes into the JSON schema shown below.

* **predict_i2b2.json** – configuration for Hugging Face sequence-tagging models.  
  * Set `model_name_or_path` to either  
    * `obi/deid_bert_i2b2`, or  
    * `obi/deid_roberta_i2b2`.

### Example JSON produced by `formatting.ipynb`

```json
{
  "text": "Physician Discharge Summary Admit date: 10/12/1982 Discharge date: 10/22/1982 Patient Information Jack Reacher, 54 y.o. male (DOB = 1/21/1928) ...",
  "meta": {"note_id": "1", "patient_id": "1"},
  "spans": []
}
```

@software{Kailas_2022_ehrdeid,
  author    = {Prajwal Kailas and Shinichi Goto and Max Homilius
               and Calum A. MacRae and Rahul C. Deo},
  title     = {EHR De-Identification Toolkit},
  year      = 2022,
  version   = {0.1.0b},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.6617957},
  url       = {https://doi.org/10.5281/zenodo.6617957}
}
