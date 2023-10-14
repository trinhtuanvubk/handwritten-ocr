
config = {
    "card_det": {"pre_process_list" : [{
                                'DetResizeForTest': {
                                    'limit_side_len': 320,
                                    'limit_type': "max"
                                }
                            }, {
                                'NormalizeImage': {
                                    'std': [0.229, 0.224, 0.225],
                                    'mean': [0.485, 0.456, 0.406],
                                    'scale': '1./255.',
                                    'order': 'hwc'
                                }
                            }, {
                                'ToCHWImage': None
                            }, {
                                'KeepKeys': {
                                    'keep_keys': ['image', 'shape']
                                }
                            }],
                "postprocess_params" : {
                    'name' : 'DBPostProcess',
                    "thresh" : 0.3,
                    "box_thresh" : 0.6,
                    "max_candidates" : 1000,
                    "unclip_ratio" : 1.5,
                    "use_dilation" : False,
                    "score_mode" : "fast",
                    "box_type": "quad"
                }
                },

    "text_det": {"pre_process_list" : [{
                                'DetResizeForTest': {
                                    'limit_side_len': 320,
                                    'limit_type': "max"
                                }
                            }, {
                                'NormalizeImage': {
                                    'std': [0.229, 0.224, 0.225],
                                    'mean': [0.485, 0.456, 0.406],
                                    'scale': '1./255.',
                                    'order': 'hwc'
                                }
                            }, {
                                'ToCHWImage': None
                            }, {
                                'KeepKeys': {
                                    'keep_keys': ['image', 'shape']
                                }
                            }],
                "postprocess_params" : {
                    'name' : 'DBPostProcess',
                    "thresh" : 0.3,
                    "box_thresh" : 0.6,
                    "max_candidates" : 1000,
                    "unclip_ratio" : 2.0,
                    "use_dilation" : False,
                    "score_mode" : "slow",
                    "box_type": "quad"
                }
                },
    "text_rec": {"postprocess_params" :{
            'name': 'CTCLabelDecode',
            "character_dict_path": '/home/ai22/Documents/VUTT/PaddleOCR/vutt_preprocess/vi_dict_full.txt',
            "use_space_char": True
        }}

}