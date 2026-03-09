Feature: API de Transacciones

  Background:
    * url 'http://localhost:8080/api'

  Scenario: Transferencia exitosa entre dos cuentas
    # Crear cuenta emisora
    Given path 'customers'
    And request { firstName: 'Emisor', lastName: 'Test', accountNumber: 'KAR-SND-01', balance: 1000.0 }
    When method POST
    Then status 200
    * def senderId = response.id

    # Crear cuenta receptora
    Given path 'customers'
    And request { firstName: 'Receptor', lastName: 'Test', accountNumber: 'KAR-RCV-01', balance: 500.0 }
    When method POST
    Then status 200
    * def receiverId = response.id

    # Realizar transferencia
    Given path 'transactions/transfer'
    And request { senderAccountNumber: 'KAR-SND-01', receiverAccountNumber: 'KAR-RCV-01', amount: 300.0 }
    When method POST
    Then status 200
    And match response.senderAccountNumber == 'KAR-SND-01'
    And match response.receiverAccountNumber == 'KAR-RCV-01'
    And match response.amount == 300.0
    And match response.id == '#notnull'
    And match response.timestamp == '#notnull'

    # Verificar saldo emisor
    Given path 'customers/' + senderId
    When method GET
    Then status 200
    And match response.balance == 700.0

    # Verificar saldo receptor
    Given path 'customers/' + receiverId
    When method GET
    Then status 200
    And match response.balance == 800.0

    # Limpiar
    Given path 'customers/' + senderId
    When method DELETE
    Then status 204
    Given path 'customers/' + receiverId
    When method DELETE
    Then status 204

  Scenario: Transferencia fallida por saldo insuficiente
    # Crear cuenta emisora con poco saldo
    Given path 'customers'
    And request { firstName: 'Pobre', lastName: 'Test', accountNumber: 'KAR-SND-02', balance: 50.0 }
    When method POST
    Then status 200
    * def senderId = response.id

    # Crear cuenta receptora
    Given path 'customers'
    And request { firstName: 'Rico', lastName: 'Test', accountNumber: 'KAR-RCV-02', balance: 500.0 }
    When method POST
    Then status 200
    * def receiverId = response.id

    # Intentar transferir mas de lo disponible
    Given path 'transactions/transfer'
    And request { senderAccountNumber: 'KAR-SND-02', receiverAccountNumber: 'KAR-RCV-02', amount: 500.0 }
    When method POST
    Then status 500

    # Limpiar
    Given path 'customers/' + senderId
    When method DELETE
    Then status 204
    Given path 'customers/' + receiverId
    When method DELETE
    Then status 204

  Scenario: Consultar historial de transacciones
    # Crear cuentas
    Given path 'customers'
    And request { firstName: 'HistA', lastName: 'Test', accountNumber: 'KAR-HST-01', balance: 1000.0 }
    When method POST
    Then status 200
    * def idA = response.id

    Given path 'customers'
    And request { firstName: 'HistB', lastName: 'Test', accountNumber: 'KAR-HST-02', balance: 500.0 }
    When method POST
    Then status 200
    * def idB = response.id

    # Realizar transferencia
    Given path 'transactions/transfer'
    And request { senderAccountNumber: 'KAR-HST-01', receiverAccountNumber: 'KAR-HST-02', amount: 100.0 }
    When method POST
    Then status 200

    # Consultar historial
    Given path 'transactions/KAR-HST-01'
    When method GET
    Then status 200
    And match response == '#array'
    And match response[0].senderAccountNumber == 'KAR-HST-01'

    # Limpiar
    Given path 'customers/' + idA
    When method DELETE
    Then status 204
    Given path 'customers/' + idB
    When method DELETE
    Then status 204
