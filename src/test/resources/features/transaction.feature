Feature: Transferencias bancarias

  Scenario: Transferencia exitosa entre dos cuentas
    Given una cuenta emisora "ACC-001" con saldo 1000.0
    And una cuenta receptora "ACC-002" con saldo 500.0
    When se transfiere 200.0 de "ACC-001" a "ACC-002"
    Then el saldo de "ACC-001" es 800.0
    And el saldo de "ACC-002" es 700.0
    And la transaccion queda registrada

  Scenario: Transferencia fallida por saldo insuficiente
    Given una cuenta emisora "ACC-001" con saldo 100.0
    And una cuenta receptora "ACC-002" con saldo 500.0
    When se transfiere 500.0 de "ACC-001" a "ACC-002"
    Then se lanza una excepcion de transferencia con mensaje "Sender Balance not enough"

  Scenario: Transferencia fallida por cuenta emisora inexistente
    Given que la cuenta "ACC-999" no existe
    When se intenta transferir 100.0 desde "ACC-999" a "ACC-002"
    Then se lanza una excepcion de transferencia con mensaje "Sender Account Number not found"

  Scenario: Consultar historial de transacciones de una cuenta
    Given una cuenta "ACC-001" con transacciones registradas
    When se consulta el historial de "ACC-001"
    Then se retorna al menos una transaccion
