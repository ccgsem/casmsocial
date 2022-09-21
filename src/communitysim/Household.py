from Place import Place
from Calendar import Calendar

class Household(Place):
    def updateInsuranceStance(self):
        self.insuranceStance = 0

    def isGettingInsurance(self, rng):
        if self.hasInsurance:
            return False

        return rng.random() > self.insuranceStance

    def step(self, calendar: Calendar):
        if calendar.isNewMonth:
            self.shopForInsurance()
        if calendar.isNewYear:
            self.reduceFuel()

    def shopForInsurance(self):
        pass

    def reduce_fuel(self):
        pass

    def purchaseInsurance(self, offers):
        pass
