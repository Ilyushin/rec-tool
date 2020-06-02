import turicreate as tc
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.response import Hit
from math import ceil
from cf_experiments_loop.hive.provider import AbstractDataProvider
from cf_experiments_loop.hive.provider.context import AbstractContext
from cf_experiments_loop.hive.provider.context import ApiDataContext
from cf_experiments_loop.hive.components.Logger import logger


class ApiDataProvider(AbstractDataProvider):
    def __init__(self, esClient: Elasticsearch):
        self._oESClient = esClient

    def supportsContext(self, context: AbstractContext) -> bool:
        return isinstance(context, ApiDataContext)

    def _doGetData(self, context: ApiDataContext) -> tc.SFrame:
        items      = list(set(context.getSourceData()['item_id']))
        nBatchSize = 10000
        nChunks    = int(ceil(len(items) / nBatchSize))
        itemChunks = np.array_split(items, nChunks)

        result = tc.SFrame()

        logger.debug(
            'Found {} items, split into {} chunks with {} items per chunk'.format(len(items), nChunks, nBatchSize)
        )

        itemChunkIndex = 0

        # @TODO Parallelize scan through multi-threading

        for itemChunk in itemChunks:
            itemChunkIndex = itemChunkIndex + 1
            logger.debug('Processing chunk {} / {}'.format(itemChunkIndex, nChunks))

            oSearch = Search(using=self._oESClient, index="trendmd") \
                .filter("ids", values=list(itemChunk)) \
                .source(fields=['campaign_id', 'journal_id', 'sponsored', 'disabled', 'title']) \
                .params(size=nBatchSize) \
                .sort('_doc')
            try:
                for hit in oSearch.scan():
                    result = result.append(self._processHit(hit))
            except:
                pass

        return result

    def _processHit(self, hit: Hit) -> tc.SFrame:
        result = tc.SFrame()
        print(hit)
        if 'id' in hit.meta is False:
            return result

        if bool(hit.disabled if 'disabled' in hit else False) is False:
            return result

        if 'campaign_id' in hit:
            localCampaignIds = hit.campaign_id

            if type(localCampaignIds) == int:
                localCampaignIds = [hit.campaign_id]

            for singleCampaignId in localCampaignIds:
                result = result.append(tc.SFrame({
                    'title'      : [str(hit.title)],
                    'item_id'    : [str(hit.meta.id)],
                    'campaign_id': [int(singleCampaignId)],
                    'journal_id' : [int(hit.journal_id)],
                    'sponsored'  : [bool(hit.sponsored if 'sponsored' in hit else False)]
                }))
        else:
            result = result.append(tc.SFrame({
                'title'      : [str(hit.title)],
                'item_id'    : [str(hit.meta.id)],
                'campaign_id': [0],
                'journal_id' : [int(hit.journal_id)],
                'sponsored'  : [bool(hit.sponsored if 'sponsored' in hit else False)]
            }))

        return result
