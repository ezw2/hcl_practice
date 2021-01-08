import numpy as np
LUT_SIZE = 2048
lut = [
	0.500000,
	0.500977,
	0.501953,
	0.502930,
	0.503906,
	0.504883,
	0.505859,
	0.506836,
	0.507812,
	0.508789,
	0.509766,
	0.510742,
	0.511719,
	0.512695,
	0.513672,
	0.514648,
	0.515625,
	0.516602,
	0.517578,
	0.518555,
	0.519531,
	0.520508,
	0.521484,
	0.522461,
	0.523438,
	0.524414,
	0.525391,
	0.526367,
	0.527344,
	0.528320,
	0.529297,
	0.530273,
	0.531250,
	0.532227,
	0.533203,
	0.534180,
	0.535156,
	0.536133,
	0.537109,
	0.538086,
	0.539062,
	0.540039,
	0.541016,
	0.541992,
	0.542969,
	0.543945,
	0.544922,
	0.545898,
	0.546875,
	0.547852,
	0.548828,
	0.549805,
	0.550781,
	0.551758,
	0.552734,
	0.553711,
	0.554688,
	0.555664,
	0.556641,
	0.557617,
	0.558594,
	0.559570,
	0.560547,
	0.561523,
	0.562500,
	0.563477,
	0.564453,
	0.565430,
	0.566406,
	0.567383,
	0.568359,
	0.569336,
	0.570312,
	0.571289,
	0.572266,
	0.573242,
	0.574219,
	0.575195,
	0.576172,
	0.577148,
	0.578125,
	0.579102,
	0.580078,
	0.581055,
	0.582031,
	0.583008,
	0.583984,
	0.584961,
	0.585938,
	0.586914,
	0.587891,
	0.588867,
	0.588867,
	0.589844,
	0.590820,
	0.591797,
	0.592773,
	0.593750,
	0.594727,
	0.595703,
	0.596680,
	0.597656,
	0.598633,
	0.599609,
	0.600586,
	0.601562,
	0.602539,
	0.603516,
	0.604492,
	0.605469,
	0.606445,
	0.607422,
	0.608398,
	0.609375,
	0.610352,
	0.611328,
	0.612305,
	0.612305,
	0.613281,
	0.614258,
	0.615234,
	0.616211,
	0.617188,
	0.618164,
	0.619141,
	0.620117,
	0.621094,
	0.622070,
	0.623047,
	0.624023,
	0.625000,
	0.625977,
	0.626953,
	0.627930,
	0.627930,
	0.628906,
	0.629883,
	0.630859,
	0.631836,
	0.632812,
	0.633789,
	0.634766,
	0.635742,
	0.636719,
	0.637695,
	0.638672,
	0.639648,
	0.640625,
	0.640625,
	0.641602,
	0.642578,
	0.643555,
	0.644531,
	0.645508,
	0.646484,
	0.647461,
	0.648438,
	0.649414,
	0.650391,
	0.650391,
	0.651367,
	0.652344,
	0.653320,
	0.654297,
	0.655273,
	0.656250,
	0.657227,
	0.658203,
	0.659180,
	0.660156,
	0.660156,
	0.661133,
	0.662109,
	0.663086,
	0.664062,
	0.665039,
	0.666016,
	0.666992,
	0.667969,
	0.667969,
	0.668945,
	0.669922,
	0.670898,
	0.671875,
	0.672852,
	0.673828,
	0.674805,
	0.674805,
	0.675781,
	0.676758,
	0.677734,
	0.678711,
	0.679688,
	0.680664,
	0.681641,
	0.681641,
	0.682617,
	0.683594,
	0.684570,
	0.685547,
	0.686523,
	0.687500,
	0.687500,
	0.688477,
	0.689453,
	0.690430,
	0.691406,
	0.692383,
	0.693359,
	0.693359,
	0.694336,
	0.695312,
	0.696289,
	0.697266,
	0.698242,
	0.699219,
	0.699219,
	0.700195,
	0.701172,
	0.702148,
	0.703125,
	0.704102,
	0.704102,
	0.705078,
	0.706055,
	0.707031,
	0.708008,
	0.708984,
	0.708984,
	0.709961,
	0.710938,
	0.711914,
	0.712891,
	0.712891,
	0.713867,
	0.714844,
	0.715820,
	0.716797,
	0.717773,
	0.717773,
	0.718750,
	0.719727,
	0.720703,
	0.721680,
	0.721680,
	0.722656,
	0.723633,
	0.724609,
	0.725586,
	0.725586,
	0.726562,
	0.727539,
	0.728516,
	0.729492,
	0.729492,
	0.730469,
	0.731445,
	0.732422,
	0.732422,
	0.733398,
	0.734375,
	0.735352,
	0.736328,
	0.736328,
	0.737305,
	0.738281,
	0.739258,
	0.739258,
	0.740234,
	0.741211,
	0.742188,
	0.743164,
	0.743164,
	0.744141,
	0.745117,
	0.746094,
	0.746094,
	0.747070,
	0.748047,
	0.749023,
	0.749023,
	0.750000,
	0.750977,
	0.751953,
	0.751953,
	0.752930,
	0.753906,
	0.754883,
	0.754883,
	0.755859,
	0.756836,
	0.756836,
	0.757812,
	0.758789,
	0.759766,
	0.759766,
	0.760742,
	0.761719,
	0.762695,
	0.762695,
	0.763672,
	0.764648,
	0.764648,
	0.765625,
	0.766602,
	0.767578,
	0.767578,
	0.768555,
	0.769531,
	0.769531,
	0.770508,
	0.771484,
	0.772461,
	0.772461,
	0.773438,
	0.774414,
	0.774414,
	0.775391,
	0.776367,
	0.776367,
	0.777344,
	0.778320,
	0.779297,
	0.779297,
	0.780273,
	0.781250,
	0.781250,
	0.782227,
	0.783203,
	0.783203,
	0.784180,
	0.785156,
	0.785156,
	0.786133,
	0.787109,
	0.787109,
	0.788086,
	0.789062,
	0.789062,
	0.790039,
	0.791016,
	0.791016,
	0.791992,
	0.792969,
	0.792969,
	0.793945,
	0.794922,
	0.794922,
	0.795898,
	0.796875,
	0.796875,
	0.797852,
	0.797852,
	0.798828,
	0.799805,
	0.799805,
	0.800781,
	0.801758,
	0.801758,
	0.802734,
	0.803711,
	0.803711,
	0.804688,
	0.804688,
	0.805664,
	0.806641,
	0.806641,
	0.807617,
	0.808594,
	0.808594,
	0.809570,
	0.809570,
	0.810547,
	0.811523,
	0.811523,
	0.812500,
	0.812500,
	0.813477,
	0.814453,
	0.814453,
	0.815430,
	0.816406,
	0.816406,
	0.817383,
	0.817383,
	0.818359,
	0.818359,
	0.819336,
	0.820312,
	0.820312,
	0.821289,
	0.821289,
	0.822266,
	0.823242,
	0.823242,
	0.824219,
	0.824219,
	0.825195,
	0.825195,
	0.826172,
	0.827148,
	0.827148,
	0.828125,
	0.828125,
	0.829102,
	0.829102,
	0.830078,
	0.831055,
	0.831055,
	0.832031,
	0.832031,
	0.833008,
	0.833008,
	0.833984,
	0.833984,
	0.834961,
	0.835938,
	0.835938,
	0.836914,
	0.836914,
	0.837891,
	0.837891,
	0.838867,
	0.838867,
	0.839844,
	0.839844,
	0.840820,
	0.841797,
	0.841797,
	0.842773,
	0.842773,
	0.843750,
	0.843750,
	0.844727,
	0.844727,
	0.845703,
	0.845703,
	0.846680,
	0.846680,
	0.847656,
	0.847656,
	0.848633,
	0.848633,
	0.849609,
	0.849609,
	0.850586,
	0.850586,
	0.851562,
	0.851562,
	0.852539,
	0.852539,
	0.853516,
	0.853516,
	0.854492,
	0.854492,
	0.855469,
	0.855469,
	0.856445,
	0.856445,
	0.857422,
	0.857422,
	0.858398,
	0.858398,
	0.859375,
	0.859375,
	0.860352,
	0.860352,
	0.861328,
	0.861328,
	0.862305,
	0.862305,
	0.863281,
	0.863281,
	0.864258,
	0.864258,
	0.864258,
	0.865234,
	0.865234,
	0.866211,
	0.866211,
	0.867188,
	0.867188,
	0.868164,
	0.868164,
	0.869141,
	0.869141,
	0.870117,
	0.870117,
	0.870117,
	0.871094,
	0.871094,
	0.872070,
	0.872070,
	0.873047,
	0.873047,
	0.874023,
	0.874023,
	0.874023,
	0.875000,
	0.875000,
	0.875977,
	0.875977,
	0.876953,
	0.876953,
	0.876953,
	0.877930,
	0.877930,
	0.878906,
	0.878906,
	0.879883,
	0.879883,
	0.879883,
	0.880859,
	0.880859,
	0.881836,
	0.881836,
	0.882812,
	0.882812,
	0.882812,
	0.883789,
	0.883789,
	0.884766,
	0.884766,
	0.884766,
	0.885742,
	0.885742,
	0.886719,
	0.886719,
	0.886719,
	0.887695,
	0.887695,
	0.888672,
	0.888672,
	0.888672,
	0.889648,
	0.889648,
	0.890625,
	0.890625,
	0.890625,
	0.891602,
	0.891602,
	0.891602,
	0.892578,
	0.892578,
	0.893555,
	0.893555,
	0.893555,
	0.894531,
	0.894531,
	0.895508,
	0.895508,
	0.895508,
	0.896484,
	0.896484,
	0.896484,
	0.897461,
	0.897461,
	0.897461,
	0.898438,
	0.898438,
	0.899414,
	0.899414,
	0.899414,
	0.900391,
	0.900391,
	0.900391,
	0.901367,
	0.901367,
	0.901367,
	0.902344,
	0.902344,
	0.902344,
	0.903320,
	0.903320,
	0.904297,
	0.904297,
	0.904297,
	0.905273,
	0.905273,
	0.905273,
	0.906250,
	0.906250,
	0.906250,
	0.907227,
	0.907227,
	0.907227,
	0.908203,
	0.908203,
	0.908203,
	0.909180,
	0.909180,
	0.909180,
	0.910156,
	0.910156,
	0.910156,
	0.911133,
	0.911133,
	0.911133,
	0.912109,
	0.912109,
	0.912109,
	0.912109,
	0.913086,
	0.913086,
	0.913086,
	0.914062,
	0.914062,
	0.914062,
	0.915039,
	0.915039,
	0.915039,
	0.916016,
	0.916016,
	0.916016,
	0.916992,
	0.916992,
	0.916992,
	0.916992,
	0.917969,
	0.917969,
	0.917969,
	0.918945,
	0.918945,
	0.918945,
	0.919922,
	0.919922,
	0.919922,
	0.919922,
	0.920898,
	0.920898,
	0.920898,
	0.921875,
	0.921875,
	0.921875,
	0.921875,
	0.922852,
	0.922852,
	0.922852,
	0.923828,
	0.923828,
	0.923828,
	0.923828,
	0.924805,
	0.924805,
	0.924805,
	0.924805,
	0.925781,
	0.925781,
	0.925781,
	0.926758,
	0.926758,
	0.926758,
	0.926758,
	0.927734,
	0.927734,
	0.927734,
	0.927734,
	0.928711,
	0.928711,
	0.928711,
	0.929688,
	0.929688,
	0.929688,
	0.929688,
	0.930664,
	0.930664,
	0.930664,
	0.930664,
	0.931641,
	0.931641,
	0.931641,
	0.931641,
	0.932617,
	0.932617,
	0.932617,
	0.932617,
	0.933594,
	0.933594,
	0.933594,
	0.933594,
	0.934570,
	0.934570,
	0.934570,
	0.934570,
	0.935547,
	0.935547,
	0.935547,
	0.935547,
	0.935547,
	0.936523,
	0.936523,
	0.936523,
	0.936523,
	0.937500,
	0.937500,
	0.937500,
	0.937500,
	0.938477,
	0.938477,
	0.938477,
	0.938477,
	0.939453,
	0.939453,
	0.939453,
	0.939453,
	0.939453,
	0.940430,
	0.940430,
	0.940430,
	0.940430,
	0.941406,
	0.941406,
	0.941406,
	0.941406,
	0.941406,
	0.942383,
	0.942383,
	0.942383,
	0.942383,
	0.942383,
	0.943359,
	0.943359,
	0.943359,
	0.943359,
	0.944336,
	0.944336,
	0.944336,
	0.944336,
	0.944336,
	0.945312,
	0.945312,
	0.945312,
	0.945312,
	0.945312,
	0.946289,
	0.946289,
	0.946289,
	0.946289,
	0.946289,
	0.947266,
	0.947266,
	0.947266,
	0.947266,
	0.947266,
	0.948242,
	0.948242,
	0.948242,
	0.948242,
	0.948242,
	0.949219,
	0.949219,
	0.949219,
	0.949219,
	0.949219,
	0.950195,
	0.950195,
	0.950195,
	0.950195,
	0.950195,
	0.950195,
	0.951172,
	0.951172,
	0.951172,
	0.951172,
	0.951172,
	0.952148,
	0.952148,
	0.952148,
	0.952148,
	0.952148,
	0.952148,
	0.953125,
	0.953125,
	0.953125,
	0.953125,
	0.953125,
	0.954102,
	0.954102,
	0.954102,
	0.954102,
	0.954102,
	0.954102,
	0.955078,
	0.955078,
	0.955078,
	0.955078,
	0.955078,
	0.955078,
	0.956055,
	0.956055,
	0.956055,
	0.956055,
	0.956055,
	0.956055,
	0.957031,
	0.957031,
	0.957031,
	0.957031,
	0.957031,
	0.957031,
	0.958008,
	0.958008,
	0.958008,
	0.958008,
	0.958008,
	0.958008,
	0.958984,
	0.958984,
	0.958984,
	0.958984,
	0.958984,
	0.958984,
	0.958984,
	0.959961,
	0.959961,
	0.959961,
	0.959961,
	0.959961,
	0.959961,
	0.960938,
	0.960938,
	0.960938,
	0.960938,
	0.960938,
	0.960938,
	0.960938,
	0.961914,
	0.961914,
	0.961914,
	0.961914,
	0.961914,
	0.961914,
	0.961914,
	0.962891,
	0.962891,
	0.962891,
	0.962891,
	0.962891,
	0.962891,
	0.962891,
	0.963867,
	0.963867,
	0.963867,
	0.963867,
	0.963867,
	0.963867,
	0.963867,
	0.964844,
	0.964844,
	0.964844,
	0.964844,
	0.964844,
	0.964844,
	0.964844,
	0.964844,
	0.965820,
	0.965820,
	0.965820,
	0.965820,
	0.965820,
	0.965820,
	0.965820,
	0.965820,
	0.966797,
	0.966797,
	0.966797,
	0.966797,
	0.966797,
	0.966797,
	0.966797,
	0.967773,
	0.967773,
	0.967773,
	0.967773,
	0.967773,
	0.967773,
	0.967773,
	0.967773,
	0.967773,
	0.968750,
	0.968750,
	0.968750,
	0.968750,
	0.968750,
	0.968750,
	0.968750,
	0.968750,
	0.969727,
	0.969727,
	0.969727,
	0.969727,
	0.969727,
	0.969727,
	0.969727,
	0.969727,
	0.969727,
	0.970703,
	0.970703,
	0.970703,
	0.970703,
	0.970703,
	0.970703,
	0.970703,
	0.970703,
	0.970703,
	0.971680,
	0.971680,
	0.971680,
	0.971680,
	0.971680,
	0.971680,
	0.971680,
	0.971680,
	0.971680,
	0.972656,
	0.972656,
	0.972656,
	0.972656,
	0.972656,
	0.972656,
	0.972656,
	0.972656,
	0.972656,
	0.973633,
	0.973633,
	0.973633,
	0.973633,
	0.973633,
	0.973633,
	0.973633,
	0.973633,
	0.973633,
	0.973633,
	0.974609,
	0.974609,
	0.974609,
	0.974609,
	0.974609,
	0.974609,
	0.974609,
	0.974609,
	0.974609,
	0.974609,
	0.974609,
	0.975586,
	0.975586,
	0.975586,
	0.975586,
	0.975586,
	0.975586,
	0.975586,
	0.975586,
	0.975586,
	0.975586,
	0.976562,
	0.976562,
	0.976562,
	0.976562,
	0.976562,
	0.976562,
	0.976562,
	0.976562,
	0.976562,
	0.976562,
	0.976562,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.977539,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.978516,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.979492,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.980469,
	0.981445,
	0.981445,
	0.981445,
	0.981445,
	0.981445,
	0.981445,
	0.981445,
	0.981445,
	0.981445,
	0.017578,
	0.017578,
	0.017578,
	0.017578,
	0.017578,
	0.017578,
	0.017578,
	0.017578,
	0.017578,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.018555,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.019531,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.020508,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.021484,
	0.022461,
	0.022461,
	0.022461,
	0.022461,
	0.022461,
	0.022461,
	0.022461,
	0.022461,
	0.022461,
	0.022461,
	0.022461,
	0.023438,
	0.023438,
	0.023438,
	0.023438,
	0.023438,
	0.023438,
	0.023438,
	0.023438,
	0.023438,
	0.023438,
	0.024414,
	0.024414,
	0.024414,
	0.024414,
	0.024414,
	0.024414,
	0.024414,
	0.024414,
	0.024414,
	0.024414,
	0.024414,
	0.025391,
	0.025391,
	0.025391,
	0.025391,
	0.025391,
	0.025391,
	0.025391,
	0.025391,
	0.025391,
	0.025391,
	0.026367,
	0.026367,
	0.026367,
	0.026367,
	0.026367,
	0.026367,
	0.026367,
	0.026367,
	0.026367,
	0.027344,
	0.027344,
	0.027344,
	0.027344,
	0.027344,
	0.027344,
	0.027344,
	0.027344,
	0.027344,
	0.028320,
	0.028320,
	0.028320,
	0.028320,
	0.028320,
	0.028320,
	0.028320,
	0.028320,
	0.028320,
	0.029297,
	0.029297,
	0.029297,
	0.029297,
	0.029297,
	0.029297,
	0.029297,
	0.029297,
	0.029297,
	0.030273,
	0.030273,
	0.030273,
	0.030273,
	0.030273,
	0.030273,
	0.030273,
	0.030273,
	0.031250,
	0.031250,
	0.031250,
	0.031250,
	0.031250,
	0.031250,
	0.031250,
	0.031250,
	0.031250,
	0.032227,
	0.032227,
	0.032227,
	0.032227,
	0.032227,
	0.032227,
	0.032227,
	0.033203,
	0.033203,
	0.033203,
	0.033203,
	0.033203,
	0.033203,
	0.033203,
	0.033203,
	0.034180,
	0.034180,
	0.034180,
	0.034180,
	0.034180,
	0.034180,
	0.034180,
	0.034180,
	0.035156,
	0.035156,
	0.035156,
	0.035156,
	0.035156,
	0.035156,
	0.035156,
	0.036133,
	0.036133,
	0.036133,
	0.036133,
	0.036133,
	0.036133,
	0.036133,
	0.037109,
	0.037109,
	0.037109,
	0.037109,
	0.037109,
	0.037109,
	0.037109,
	0.038086,
	0.038086,
	0.038086,
	0.038086,
	0.038086,
	0.038086,
	0.038086,
	0.039062,
	0.039062,
	0.039062,
	0.039062,
	0.039062,
	0.039062,
	0.040039,
	0.040039,
	0.040039,
	0.040039,
	0.040039,
	0.040039,
	0.040039,
	0.041016,
	0.041016,
	0.041016,
	0.041016,
	0.041016,
	0.041016,
	0.041992,
	0.041992,
	0.041992,
	0.041992,
	0.041992,
	0.041992,
	0.042969,
	0.042969,
	0.042969,
	0.042969,
	0.042969,
	0.042969,
	0.043945,
	0.043945,
	0.043945,
	0.043945,
	0.043945,
	0.043945,
	0.044922,
	0.044922,
	0.044922,
	0.044922,
	0.044922,
	0.044922,
	0.045898,
	0.045898,
	0.045898,
	0.045898,
	0.045898,
	0.046875,
	0.046875,
	0.046875,
	0.046875,
	0.046875,
	0.046875,
	0.047852,
	0.047852,
	0.047852,
	0.047852,
	0.047852,
	0.048828,
	0.048828,
	0.048828,
	0.048828,
	0.048828,
	0.048828,
	0.049805,
	0.049805,
	0.049805,
	0.049805,
	0.049805,
	0.050781,
	0.050781,
	0.050781,
	0.050781,
	0.050781,
	0.051758,
	0.051758,
	0.051758,
	0.051758,
	0.051758,
	0.052734,
	0.052734,
	0.052734,
	0.052734,
	0.052734,
	0.053711,
	0.053711,
	0.053711,
	0.053711,
	0.053711,
	0.054688,
	0.054688,
	0.054688,
	0.054688,
	0.054688,
	0.055664,
	0.055664,
	0.055664,
	0.055664,
	0.056641,
	0.056641,
	0.056641,
	0.056641,
	0.056641,
	0.057617,
	0.057617,
	0.057617,
	0.057617,
	0.057617,
	0.058594,
	0.058594,
	0.058594,
	0.058594,
	0.059570,
	0.059570,
	0.059570,
	0.059570,
	0.059570,
	0.060547,
	0.060547,
	0.060547,
	0.060547,
	0.061523,
	0.061523,
	0.061523,
	0.061523,
	0.062500,
	0.062500,
	0.062500,
	0.062500,
	0.063477,
	0.063477,
	0.063477,
	0.063477,
	0.063477,
	0.064453,
	0.064453,
	0.064453,
	0.064453,
	0.065430,
	0.065430,
	0.065430,
	0.065430,
	0.066406,
	0.066406,
	0.066406,
	0.066406,
	0.067383,
	0.067383,
	0.067383,
	0.067383,
	0.068359,
	0.068359,
	0.068359,
	0.068359,
	0.069336,
	0.069336,
	0.069336,
	0.069336,
	0.070312,
	0.070312,
	0.070312,
	0.071289,
	0.071289,
	0.071289,
	0.071289,
	0.072266,
	0.072266,
	0.072266,
	0.072266,
	0.073242,
	0.073242,
	0.073242,
	0.074219,
	0.074219,
	0.074219,
	0.074219,
	0.075195,
	0.075195,
	0.075195,
	0.075195,
	0.076172,
	0.076172,
	0.076172,
	0.077148,
	0.077148,
	0.077148,
	0.077148,
	0.078125,
	0.078125,
	0.078125,
	0.079102,
	0.079102,
	0.079102,
	0.079102,
	0.080078,
	0.080078,
	0.080078,
	0.081055,
	0.081055,
	0.081055,
	0.082031,
	0.082031,
	0.082031,
	0.082031,
	0.083008,
	0.083008,
	0.083008,
	0.083984,
	0.083984,
	0.083984,
	0.084961,
	0.084961,
	0.084961,
	0.085938,
	0.085938,
	0.085938,
	0.086914,
	0.086914,
	0.086914,
	0.086914,
	0.087891,
	0.087891,
	0.087891,
	0.088867,
	0.088867,
	0.088867,
	0.089844,
	0.089844,
	0.089844,
	0.090820,
	0.090820,
	0.090820,
	0.091797,
	0.091797,
	0.091797,
	0.092773,
	0.092773,
	0.092773,
	0.093750,
	0.093750,
	0.093750,
	0.094727,
	0.094727,
	0.094727,
	0.095703,
	0.095703,
	0.096680,
	0.096680,
	0.096680,
	0.097656,
	0.097656,
	0.097656,
	0.098633,
	0.098633,
	0.098633,
	0.099609,
	0.099609,
	0.099609,
	0.100586,
	0.100586,
	0.101562,
	0.101562,
	0.101562,
	0.102539,
	0.102539,
	0.102539,
	0.103516,
	0.103516,
	0.103516,
	0.104492,
	0.104492,
	0.105469,
	0.105469,
	0.105469,
	0.106445,
	0.106445,
	0.107422,
	0.107422,
	0.107422,
	0.108398,
	0.108398,
	0.108398,
	0.109375,
	0.109375,
	0.110352,
	0.110352,
	0.110352,
	0.111328,
	0.111328,
	0.112305,
	0.112305,
	0.112305,
	0.113281,
	0.113281,
	0.114258,
	0.114258,
	0.114258,
	0.115234,
	0.115234,
	0.116211,
	0.116211,
	0.116211,
	0.117188,
	0.117188,
	0.118164,
	0.118164,
	0.119141,
	0.119141,
	0.119141,
	0.120117,
	0.120117,
	0.121094,
	0.121094,
	0.122070,
	0.122070,
	0.122070,
	0.123047,
	0.123047,
	0.124023,
	0.124023,
	0.125000,
	0.125000,
	0.125000,
	0.125977,
	0.125977,
	0.126953,
	0.126953,
	0.127930,
	0.127930,
	0.128906,
	0.128906,
	0.128906,
	0.129883,
	0.129883,
	0.130859,
	0.130859,
	0.131836,
	0.131836,
	0.132812,
	0.132812,
	0.133789,
	0.133789,
	0.134766,
	0.134766,
	0.134766,
	0.135742,
	0.135742,
	0.136719,
	0.136719,
	0.137695,
	0.137695,
	0.138672,
	0.138672,
	0.139648,
	0.139648,
	0.140625,
	0.140625,
	0.141602,
	0.141602,
	0.142578,
	0.142578,
	0.143555,
	0.143555,
	0.144531,
	0.144531,
	0.145508,
	0.145508,
	0.146484,
	0.146484,
	0.147461,
	0.147461,
	0.148438,
	0.148438,
	0.149414,
	0.149414,
	0.150391,
	0.150391,
	0.151367,
	0.151367,
	0.152344,
	0.152344,
	0.153320,
	0.153320,
	0.154297,
	0.154297,
	0.155273,
	0.155273,
	0.156250,
	0.156250,
	0.157227,
	0.157227,
	0.158203,
	0.159180,
	0.159180,
	0.160156,
	0.160156,
	0.161133,
	0.161133,
	0.162109,
	0.162109,
	0.163086,
	0.163086,
	0.164062,
	0.165039,
	0.165039,
	0.166016,
	0.166016,
	0.166992,
	0.166992,
	0.167969,
	0.167969,
	0.168945,
	0.169922,
	0.169922,
	0.170898,
	0.170898,
	0.171875,
	0.171875,
	0.172852,
	0.173828,
	0.173828,
	0.174805,
	0.174805,
	0.175781,
	0.175781,
	0.176758,
	0.177734,
	0.177734,
	0.178711,
	0.178711,
	0.179688,
	0.180664,
	0.180664,
	0.181641,
	0.181641,
	0.182617,
	0.182617,
	0.183594,
	0.184570,
	0.184570,
	0.185547,
	0.186523,
	0.186523,
	0.187500,
	0.187500,
	0.188477,
	0.189453,
	0.189453,
	0.190430,
	0.190430,
	0.191406,
	0.192383,
	0.192383,
	0.193359,
	0.194336,
	0.194336,
	0.195312,
	0.195312,
	0.196289,
	0.197266,
	0.197266,
	0.198242,
	0.199219,
	0.199219,
	0.200195,
	0.201172,
	0.201172,
	0.202148,
	0.202148,
	0.203125,
	0.204102,
	0.204102,
	0.205078,
	0.206055,
	0.206055,
	0.207031,
	0.208008,
	0.208008,
	0.208984,
	0.209961,
	0.209961,
	0.210938,
	0.211914,
	0.211914,
	0.212891,
	0.213867,
	0.213867,
	0.214844,
	0.215820,
	0.215820,
	0.216797,
	0.217773,
	0.217773,
	0.218750,
	0.219727,
	0.219727,
	0.220703,
	0.221680,
	0.222656,
	0.222656,
	0.223633,
	0.224609,
	0.224609,
	0.225586,
	0.226562,
	0.226562,
	0.227539,
	0.228516,
	0.229492,
	0.229492,
	0.230469,
	0.231445,
	0.231445,
	0.232422,
	0.233398,
	0.234375,
	0.234375,
	0.235352,
	0.236328,
	0.236328,
	0.237305,
	0.238281,
	0.239258,
	0.239258,
	0.240234,
	0.241211,
	0.242188,
	0.242188,
	0.243164,
	0.244141,
	0.244141,
	0.245117,
	0.246094,
	0.247070,
	0.247070,
	0.248047,
	0.249023,
	0.250000,
	0.250000,
	0.250977,
	0.251953,
	0.252930,
	0.252930,
	0.253906,
	0.254883,
	0.255859,
	0.255859,
	0.256836,
	0.257812,
	0.258789,
	0.259766,
	0.259766,
	0.260742,
	0.261719,
	0.262695,
	0.262695,
	0.263672,
	0.264648,
	0.265625,
	0.266602,
	0.266602,
	0.267578,
	0.268555,
	0.269531,
	0.269531,
	0.270508,
	0.271484,
	0.272461,
	0.273438,
	0.273438,
	0.274414,
	0.275391,
	0.276367,
	0.277344,
	0.277344,
	0.278320,
	0.279297,
	0.280273,
	0.281250,
	0.281250,
	0.282227,
	0.283203,
	0.284180,
	0.285156,
	0.286133,
	0.286133,
	0.287109,
	0.288086,
	0.289062,
	0.290039,
	0.290039,
	0.291016,
	0.291992,
	0.292969,
	0.293945,
	0.294922,
	0.294922,
	0.295898,
	0.296875,
	0.297852,
	0.298828,
	0.299805,
	0.299805,
	0.300781,
	0.301758,
	0.302734,
	0.303711,
	0.304688,
	0.305664,
	0.305664,
	0.306641,
	0.307617,
	0.308594,
	0.309570,
	0.310547,
	0.311523,
	0.311523,
	0.312500,
	0.313477,
	0.314453,
	0.315430,
	0.316406,
	0.317383,
	0.317383,
	0.318359,
	0.319336,
	0.320312,
	0.321289,
	0.322266,
	0.323242,
	0.324219,
	0.324219,
	0.325195,
	0.326172,
	0.327148,
	0.328125,
	0.329102,
	0.330078,
	0.331055,
	0.331055,
	0.332031,
	0.333008,
	0.333984,
	0.334961,
	0.335938,
	0.336914,
	0.337891,
	0.338867,
	0.338867,
	0.339844,
	0.340820,
	0.341797,
	0.342773,
	0.343750,
	0.344727,
	0.345703,
	0.346680,
	0.347656,
	0.348633,
	0.348633,
	0.349609,
	0.350586,
	0.351562,
	0.352539,
	0.353516,
	0.354492,
	0.355469,
	0.356445,
	0.357422,
	0.358398,
	0.358398,
	0.359375,
	0.360352,
	0.361328,
	0.362305,
	0.363281,
	0.364258,
	0.365234,
	0.366211,
	0.367188,
	0.368164,
	0.369141,
	0.370117,
	0.371094,
	0.371094,
	0.372070,
	0.373047,
	0.374023,
	0.375000,
	0.375977,
	0.376953,
	0.377930,
	0.378906,
	0.379883,
	0.380859,
	0.381836,
	0.382812,
	0.383789,
	0.384766,
	0.385742,
	0.386719,
	0.386719,
	0.387695,
	0.388672,
	0.389648,
	0.390625,
	0.391602,
	0.392578,
	0.393555,
	0.394531,
	0.395508,
	0.396484,
	0.397461,
	0.398438,
	0.399414,
	0.400391,
	0.401367,
	0.402344,
	0.403320,
	0.404297,
	0.405273,
	0.406250,
	0.407227,
	0.408203,
	0.409180,
	0.410156,
	0.410156,
	0.411133,
	0.412109,
	0.413086,
	0.414062,
	0.415039,
	0.416016,
	0.416992,
	0.417969,
	0.418945,
	0.419922,
	0.420898,
	0.421875,
	0.422852,
	0.423828,
	0.424805,
	0.425781,
	0.426758,
	0.427734,
	0.428711,
	0.429688,
	0.430664,
	0.431641,
	0.432617,
	0.433594,
	0.434570,
	0.435547,
	0.436523,
	0.437500,
	0.438477,
	0.439453,
	0.440430,
	0.441406,
	0.442383,
	0.443359,
	0.444336,
	0.445312,
	0.446289,
	0.447266,
	0.448242,
	0.449219,
	0.450195,
	0.451172,
	0.452148,
	0.453125,
	0.454102,
	0.455078,
	0.456055,
	0.457031,
	0.458008,
	0.458984,
	0.459961,
	0.460938,
	0.461914,
	0.462891,
	0.463867,
	0.464844,
	0.465820,
	0.466797,
	0.467773,
	0.468750,
	0.469727,
	0.470703,
	0.471680,
	0.472656,
	0.473633,
	0.474609,
	0.475586,
	0.476562,
	0.477539,
	0.478516,
	0.479492,
	0.480469,
	0.481445,
	0.482422,
	0.483398,
	0.484375,
	0.485352,
	0.486328,
	0.487305,
	0.488281,
	0.489258,
	0.490234,
	0.491211,
	0.492188,
	0.493164,
	0.494141,
	0.495117,
	0.496094,
	0.497070,
	0.498047,
	0.499023
        ]
        

