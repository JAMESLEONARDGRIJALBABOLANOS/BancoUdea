package com.udea.bancoudea.karate;

import com.intuit.karate.junit5.Karate;

class KarateRunnerTest {

    @Karate.Test
    Karate testCustomers() {
        return Karate.run("customers").relativeTo(getClass());
    }

    @Karate.Test
    Karate testTransactions() {
        return Karate.run("transactions").relativeTo(getClass());
    }
}
