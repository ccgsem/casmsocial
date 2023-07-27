from repast4py.space import DiscretePoint as dpt
from scipy.stats import triang, norm
from .Household import Household
from .Parameters import Parameters


class InsuranceProvider(object):
    def __init__(self, insurerId: int):
        self.id = insurerId
        self.riskTolerance = None # low, med, high
        self.threshold = 0 # arbitrarily assigning a threshold of 100 homes in a census block to each insurance provider (ie. rational is you don't want to insure all homes in a block because if it burns down it is catastrophic)
        self.householdsInsured = [] # array of the households insured for each providers

    def set_risk_tolerance(self, rng):
        
        tolerance = rng.random()

        if tolerance < .33:
            self.riskTolerance = "low"
        elif tolerance < .66:
            self.riskTolerance = "med"
        else:
            self.riskTolerance = "high"

    def set_threshold(self, rng):

        threshold = rng.random() / 20 # scale the threshold down by 20 for some variety for each individual provider

        if self.riskTolerance == "low":
            self.threshold = Parameters.neighborhoodInsuredBelowThrehold - threshold
        elif self.riskTolerance == "low":
            self.threshold = Parameters.neighborhoodInsuredBelowThrehold
        else:
            self.threshold = Parameters.neighborhoodInsuredBelowThrehold + threshold


    def compute_premium(self, hh: Household):
        # crps: conditional risk to potential structures (ie. if there is a fire here and a home here, what is the chance it is burned down?)
        # fuel_crps: the probability of structure loss given fuel reductions
        # bp: burn probability (ie. what is the chance a fire will be here)
        fireRisk = (hh.fireCRiskFuel + hh.fireCRisk)/2 * hh.fireBP

        # Adjust the fireRisk for the insurer's risk tolerance
        # Again, would love some evidence for the adjustment factor
        adjustmentFactor = norm.rvs(loc=.0001, scale=.00001, size=1, random_state=1)
        if self.riskTolerance == "low":
            fireRisk = fireRisk + adjustmentFactor
        elif self.riskTolerance == "high":
            fireRisk = fireRisk - adjustmentFactor

        # Premium calculation
        # Since there is a $200,000 dwelling replacement cost limit and $160,000 limit on the contents, I am creating a "replacmentCost" variable to capture the total potential payout from a policy

        # Structure

        # The US Army Corps of Engineers estimates that the replacement value of a structure depreciates by 1% each year for the first 20 years after it is built/renovated.
        # After 20 years, it is assumed that regular maintenance will keep the replacement value at 80% of their original value.
        # I am using the "EffectiveYear" column as the year when the 20 year counter should stop.
        replacementCostStructure = 0
        depreciationPercentage = 2022 - hh.structYearBuilt
        if depreciationPercentage > 20:
            depreciationPercentage = 20
        depreciationPercentage = depreciationPercentage / 100

        # adjust the building value based on the depreciation precentage
        structAdjVal = hh.structVal * (1 - depreciationPercentage)
        if structAdjVal > 200000:
            replacementCostStructure = 200000
        else:
            replacementCostStructure = structAdjVal

        # Contents

        # For the contents, the US Army Corps of Engineers estimates that the content-to-structure value ratio is .5 for all residential homes
        # This means that the estimted value of the contents in each residential home is half the value of the structure
        replacementCostContents = 0
        if structAdjVal * .5 > 160000:
            replacementCostContents = 160000
        else:
            replacementCostContents = structAdjVal * .5

        # Army Corps of Engineers NSI Technical Documentation - Structure Valuation and Occupancy Type
        # https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/latest/technical-documentation#id-.TechnicalDocumentationv2022-StructureValuation

        # Calculate the premium value
        # I am creating a random variable given the aforementioned distribution around each fire risk point estimate
        # I would like some real substance/evidence behind the .0005 standard deviation number, but I don't have that
        assessedRisk = norm.rvs(loc=fireRisk, scale=.0005, size=1, random_state=1) # Create the risk distribution
        # risk_series = [x if x >=0 else 0 for x in dist]
        # cost_series =  [x * (replacementCostContents + replacementCostStructure) if x >= 0 else 0 for x in dist] 
        if assessedRisk < 0:
            assessedRisk = 0
        premium = assessedRisk * (replacementCostContents + replacementCostStructure)

        # Scale the premium up to model the full population
        # TO DO: IMPLEMENT ME

        return premium

    def provide_insurance(self, hh: Household):

        # Decision Variables:
        threshold = "neighborhoodInsuredBelowThreshold"
        # TO DO: IMPLEMENT ME
        # I have no idea how to do this... need help, but here is the pseudo-code
        # Measure if the insurance provider already has too many homes in the census block to insure one more
        # number_of_homes_in_block = hh.location.count_number_of_homes()
        # if number_of_homes_insured_in_block / number_of_homes_in_block > self.threshold:
        #   threshold = "neighborhoodInsuredAboveThreshold"

        # Create decision rules tuple
        query = (hh.fireCRisk, hh.structureLossProbability, threshold, self.riskTolerance)

        # Query Decision Rules Dict for the resulting action
        provideInsurance = decisionRuleMap['provide_insurance'].get(query)

        if provideInsurance == None:
            return None # Some error handling here, I don't think this should happen

        pProvide = rng.random()

        if provideInsurance == "provideInsPH" and pProvide < Parameters.provideInsPH:
            premium = self.compute_premium(hh)
            return premium
        elif provideInsurance == "provideInsPM" and pProvide < Parameters.provideInsPM:
            premium = self.compute_premium(hh)
            return premium
        elif provideInsurance == "provideInsPL" and pProvide < Parameters.provideInsPL:
            premium = self.compute_premium(hh)
            return premium
        else:
            return None # No insurance for you
