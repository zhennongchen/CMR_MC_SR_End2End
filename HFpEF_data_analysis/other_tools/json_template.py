import os
import json

def json_template(contour_point_positions, color_code):
    data = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
    "markups": [
        {
            "type": "ClosedCurve",
            "coordinateSystem": "LPS",
            "coordinateUnits": "mm",
            "locked": False,
            "fixedNumberOfControlPoints": False,
            "labelFormat": "%N-%d",
            "lastUsedControlPointNumber": len(contour_point_positions),
            "controlPoints": [
                {
                    "id": str(i+1),
                    "label": "CC_{}".format(i+1),
                    "description": "",
                    "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                    "position": [float(xx) for xx in contour_point_positions[i]],
                    "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                    "selected": True,
                    "locked": False,
                    "visibility": True,
                    "positionStatus": "defined"
                } for i in range(len(contour_point_positions))
            ],
            "measurements": [
                {
                    "name": "length",
                    "enabled": False,
                    "units": "mm",
                    "printFormat": "%-#4.4g%s"
                },
                {
                    "name": "curvature mean",
                    "enabled": False,
                    "printFormat": "%5.3f %s"
                },
                {
                    "name": "curvature max",
                    "enabled": False,
                    "printFormat": "%5.3f %s"
                },
                {
                    "name": "area",
                    "enabled": False,
                    "units": "cm2",
                    "printFormat": "%-#4.4g%s"
                }
            ],
            "display": {
                "visibility": True,
                "opacity": 0.5,
                "color": [0.4, 1.0, 0.0],
                "selectedColor": color_code,
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": True,
                "pointLabelsVisibility": False,
                "textScale": 2.2,
                "glyphType": "Sphere3D",
                "glyphScale": 2.4000000000000005,
                "glyphSize": 5.0,
                "useGlyphScale": True,
                "sliceProjection": False,
                "sliceProjectionUseFiducialColor": True,
                "sliceProjectionOutlinedBehindSlicePlane": False,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": False,
                "translationHandleVisibility": True,
                "rotationHandleVisibility": True,
                "scaleHandleVisibility": True,
                "interactionHandleScale": 3.0,
                "snapMode": "toVisibleSurface"
                }
            }
        ]
    }
    return data