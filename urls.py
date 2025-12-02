from django.urls import path
from .views import AlphaMinerAPI, UploadCSVView , HeuristicMinerAPI
urlpatterns = [
    path('uploadCSV/', UploadCSVView.as_view(), name='upload-csv'),
    path('alphaMiner/', AlphaMinerAPI.as_view(), name='alpha-miner'),
    path('heuristicMiner/', HeuristicMinerAPI.as_view(), name='heuristic_miner'),
]
