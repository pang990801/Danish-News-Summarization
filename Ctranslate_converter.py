from ctranslate2.converters import TransformersConverter

# nllb_3B_model = TransformersConverter("facebook/nllb-200-3.3B")
# nllb_600M_model = TransformersConverter("facebook/nllb-200-distilled-600m")
# opus_da_en_model = TransformersConverter("Helsinki-NLP/opus-mt-da-en")
opus_en_da_model = TransformersConverter("Helsinki-NLP/opus-mt-en-da")

#output_dir_nllb_3B = "models/nllb-200-3.3B_ct2"
#output_dir_nllb_600M = "models/nllb-200-distilled-600m_ct2"
#output_dir_opus_da_en = "models/opus-mt-da-en_ct2"
output_dir_opus_en_da = "models/opus-mt-en-da_ct2"

# nllb_3B_model.convert(output_dir_nllb_3B)
# nllb_600M_model.convert(output_dir_nllb_600M)
# opus_da_en_model.convert(output_dir_opus_da_en)
opus_en_da_model.convert(output_dir_opus_en_da)
